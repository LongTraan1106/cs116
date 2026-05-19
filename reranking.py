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
