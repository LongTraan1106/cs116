import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


def positive_sampling(transaction_lf, q_val):
    print(">> Positive Sampling (Ground Truth Purchases)...")
    val_lf = transaction_lf.filter(pl.sql_expr(q_val))
    val_cust_ids = val_lf.select("customer_id").unique().collect()["customer_id"].to_list()

    pos_df = (
        val_lf
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("target"))
        .collect()
    )
    return pos_df, val_cust_ids


def negative_sampling(candidates_df, pos_df, cfg):
    """Sample split-specific hard negatives, scaled by positives per user."""
    print(">> Negative Sampling (From Candidates)...")

    if isinstance(pos_df, pd.DataFrame):
        pos_df = pl.from_pandas(pos_df)

    n_neg_per_pos = cfg.get("n_neg_per_pos", 5)
    min_neg_per_user = cfg.get("min_neg_per_user", 5)
    max_neg_per_user = cfg.get("max_neg_per_user", 50)
    invalid_items = {"(not set)", "not set", "", None}

    pos_keys = (
        pos_df
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        ])
        .filter(~pl.col("item_id").is_in(invalid_items))
        .unique()
    )

    pos_counts = (
        pos_keys
        .group_by("customer_id")
        .len()
        .with_columns(
            pl.min_horizontal([
                pl.lit(max_neg_per_user),
                pl.max_horizontal([
                    pl.lit(min_neg_per_user),
                    pl.col("len") * n_neg_per_pos,
                ]),
            ]).cast(pl.UInt32).alias("n_neg_user")
        )
        .select(["customer_id", "n_neg_user"])
    )

    candidate_keys = (
        candidates_df
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        ])
        .filter(~pl.col("item_id").is_in(invalid_items) & pl.col("item_id").is_not_null())
        .unique()
    )

    available_neg = (
        candidate_keys
        .join(pos_keys, on=["customer_id", "item_id"], how="anti")
        .join(pos_counts, on="customer_id", how="inner")
    )
    available_count = available_neg.height

    if available_neg.height == 0:
        print("   -> No available negatives after anti-join.")
        return pl.DataFrame(
            {"customer_id": [], "item_id": [], "target": []},
            schema={
                "customer_id": pos_df.schema["customer_id"],
                "item_id": pos_df.schema["item_id"],
                "target": pl.Int64,
            },
        )

    rng = np.random.default_rng(42)
    available_neg = (
        available_neg
        .with_columns(pl.Series("_rand", rng.random(available_neg.height)))
        .sort(["customer_id", "_rand"])
        .with_columns(pl.col("item_id").cum_count().over("customer_id").alias("_rn"))
        .filter(pl.col("_rn") <= pl.col("n_neg_user"))
    )

    final_neg_df = available_neg.select([
        pl.col("customer_id").cast(pos_df.schema["customer_id"], strict=False),
        pl.col("item_id").cast(pos_df.schema["item_id"], strict=False),
        pl.lit(0, dtype=pl.Int64).alias("target"),
    ])

    print(
        f"   -> Pos users: {pos_counts.height}, "
        f"available negatives: {available_count}, sampled negatives: {final_neg_df.height}"
    )
    return final_neg_df


def get_prediction_expr(model, feature_cols):
    return None


def _build_group_vector(df_frame):
    sorted_frame = df_frame.sort(["customer_id", "item_id"])
    group = (
        sorted_frame
        .group_by("customer_id", maintain_order=True)
        .len()["len"]
        .to_numpy()
    )
    return sorted_frame, group


def train_model(df_train, feature_cols, model_name, cfg, df_val=None, early_stopping_rounds=50):
    print(f">> Training Model: {model_name} (LightGBM Ranker)...")

    print("   -> Sorting training rows by customer_id/item_id...")
    df_train = df_train.sort(["customer_id", "item_id"])

    print("   -> Converting Training Data to Numpy (Float32)...")
    try:
        X = df_train.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
        y = df_train.select("target").to_numpy().ravel()
    except Exception:
        X = df_train.select(feature_cols).fill_null(0).to_pandas().values.astype(np.float32)
        y = df_train.select("target").to_pandas().values.ravel()

    print("   -> Building group (by customer_id)...")
    try:
        _, group = _build_group_vector(df_train)
    except Exception:
        group = df_train.to_pandas().groupby("customer_id").size().values

    print(f"   -> Total groups: {len(group)}")

    eval_set = None
    eval_group = None
    if df_val is not None and df_val.height > 0:
        print("   -> Preparing validation set for early stopping...")
        df_val = df_val.sort(["customer_id", "item_id"])
        try:
            X_val = df_val.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
            y_val = df_val.select("target").to_numpy().ravel()
        except Exception:
            X_val = df_val.select(feature_cols).fill_null(0).to_pandas().values.astype(np.float32)
            y_val = df_val.select("target").to_pandas().values.ravel()

        try:
            _, group_val = _build_group_vector(df_val)
        except Exception:
            group_val = df_val.to_pandas().groupby("customer_id").size().values

        eval_set = [(X_val, y_val)]
        eval_group = [group_val]
        print(f"   -> Validation shape: {X_val.shape}, groups: {len(group_val)}")

    print("   -> Configuring LightGBM Ranker (CPU)...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=cfg.get("n_estimators", 2000),
        learning_rate=cfg.get("learning_rate", 0.03),
        num_leaves=cfg.get("num_leaves", 63),
        min_child_samples=cfg.get("min_child_samples", 30),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print(f"   -> Fitting ranker on {X.shape} matrix...")
    fit_kwargs = {"group": group}
    if eval_set is not None:
        fit_kwargs.update({
            "eval_set": eval_set,
            "eval_group": eval_group,
            "eval_metric": "ndcg",
            "callbacks": [
                lgb.early_stopping(early_stopping_rounds, verbose=True),
                lgb.log_evaluation(25),
            ],
        })
    else:
        fit_kwargs.update({"callbacks": [lgb.log_evaluation(25)]})

    model.fit(X, y, **fit_kwargs)

    print("\nMODEL BIAS (Baseline Score Reference):")
    preds = model.predict(X)
    print(f"   -> Mean prediction score: {float(np.mean(preds)):.6f}")

    print("\nFEATURE IMPORTANCE (LightGBM Ranker):")
    importance = model.booster_.feature_importance(importance_type="gain")
    imp_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importance})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )
    total_importance = imp_df["Importance"].sum()
    imp_df["Importance_norm"] = imp_df["Importance"] / total_importance if total_importance else 0.0
    print(imp_df.to_string(index=False, formatters={"Importance_norm": "{:.4f}".format}))

    print("\nDEBUG NOTES:")
    print(" - Feature < 1% can be considered for pruning.")
    print(" - A dominant feature is a leakage smell; inspect it before trusting gains.")
    print(" - Higher bias with lower precision usually means the candidate pool is too broad.")
    print("=" * 60)

    return model


def train_catboost_ranker(df_train, feature_cols, cfg, df_val=None):
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as e:
        raise ImportError(
            "CatBoost is required for v15_lgbm_catboost_blend. "
            "Install it with: pip install 'catboost>=1.2.5'"
        ) from e

    print(">> Training Model: CatBoostRanker...")
    print("   -> Sorting training rows by customer_id/item_id...")
    df_train = df_train.sort(["customer_id", "item_id"])

    print("   -> Converting CatBoost training data to Float32...")
    X_train = df_train.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
    y_train = df_train.select("target").to_numpy().ravel()
    group_id_train = df_train.get_column("customer_id").cast(pl.Utf8).to_list()

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=group_id_train,
        feature_names=feature_cols,
    )

    eval_set = None
    if df_val is not None and df_val.height > 0:
        print("   -> Preparing CatBoost validation pool...")
        df_val = df_val.sort(["customer_id", "item_id"])
        X_val = df_val.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
        y_val = df_val.select("target").to_numpy().ravel()
        group_id_val = df_val.get_column("customer_id").cast(pl.Utf8).to_list()
        eval_set = Pool(
            data=X_val,
            label=y_val,
            group_id=group_id_val,
            feature_names=feature_cols,
        )

    cat_model = CatBoostRanker(
        loss_function=cfg.get("cat_loss_function", "YetiRank"),
        eval_metric=cfg.get("cat_eval_metric", "NDCG:top=10"),
        iterations=cfg.get("cat_iterations", 1200),
        learning_rate=cfg.get("cat_learning_rate", 0.05),
        depth=cfg.get("cat_depth", 8),
        l2_leaf_reg=cfg.get("cat_l2_leaf_reg", 5.0),
        random_seed=cfg.get("cat_random_seed", 42),
        od_type="Iter",
        od_wait=cfg.get("cat_od_wait", 50),
        thread_count=-1,
        verbose=100,
        allow_writing_files=False,
    )

    cat_model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)

    try:
        print(f"   -> CatBoost best iteration: {cat_model.get_best_iteration()}")
    except Exception:
        pass

    try:
        importance = cat_model.get_feature_importance()
        imp_df = (
            pd.DataFrame({"Feature": feature_cols, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .head(30)
        )
        print("\nFEATURE IMPORTANCE (CatBoost Ranker Top 30):")
        print(imp_df.to_string(index=False))
    except Exception as e:
        print(f"   Warning: CatBoost feature importance unavailable: {e}")

    return cat_model


def predict_catboost(cat_model, df_features, feature_cols):
    X = df_features.select(feature_cols).fill_null(0).cast(pl.Float32).to_numpy()
    scores = np.asarray(cat_model.predict(X), dtype=np.float32)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_rank_blend_scores(
    df,
    lgb_col="pred_score",
    cat_col="cat_score",
    w_lgb=0.65,
    output_col="final_score",
):
    return (
        df
        .with_columns([
            pl.col(lgb_col).rank("ordinal", descending=True).over("customer_id").alias("_lgb_rank"),
            pl.col(cat_col).rank("ordinal", descending=True).over("customer_id").alias("_cat_rank"),
            pl.len().over("customer_id").alias("_candidate_count"),
        ])
        .with_columns([
            (
                (pl.col("_candidate_count") - pl.col("_lgb_rank") + 1)
                / pl.col("_candidate_count")
            ).alias("_lgb_rank_norm"),
            (
                (pl.col("_candidate_count") - pl.col("_cat_rank") + 1)
                / pl.col("_candidate_count")
            ).alias("_cat_rank_norm"),
        ])
        .with_columns(
            (
                (w_lgb * pl.col("_lgb_rank_norm"))
                + ((1.0 - w_lgb) * pl.col("_cat_rank_norm"))
            ).cast(pl.Float32).alias(output_col)
        )
        .drop([
            "_lgb_rank",
            "_cat_rank",
            "_candidate_count",
            "_lgb_rank_norm",
            "_cat_rank_norm",
        ])
    )


def grid_search_blend_weight(df_val_pred, pos_val_df, weights, eval_module, k=10):
    print("\n>> Blend Weight Grid Search (rank-normalized)")
    best_w = None
    best_metrics = None
    best_key = None

    print("   w_lgb | precision@10 | map | mrr | ndcg | recall@10 | iou")
    for w in weights:
        blended = add_rank_blend_scores(
            df_val_pred,
            lgb_col="pred_score",
            cat_col="cat_score",
            w_lgb=float(w),
            output_col="final_score",
        ).with_columns(pl.col("final_score").alias("pred_score"))

        metrics = eval_module.evaluate_predictions_df(blended, pos_val_df, k=k)
        metric_key = (
            metrics.get("precision@k", 0.0),
            metrics.get("map", 0.0),
            metrics.get("mrr", 0.0),
            metrics.get("ndcg@k", 0.0),
        )
        print(
            f"   {float(w):.2f} | "
            f"{metrics.get('precision@k', 0.0):.6f} | "
            f"{metrics.get('map', 0.0):.6f} | "
            f"{metrics.get('mrr', 0.0):.6f} | "
            f"{metrics.get('ndcg@k', 0.0):.6f} | "
            f"{metrics.get('recall@k', 0.0):.6f} | "
            f"{metrics.get('iou', 0.0):.6f}"
        )

        if best_key is None or metric_key > best_key:
            best_key = metric_key
            best_w = float(w)
            best_metrics = metrics

    print(f">> Selected blend weight: w_lgb={best_w:.2f}")
    return best_w, best_metrics
