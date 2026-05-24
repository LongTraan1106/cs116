from config import load_config
import get_candidates as gc_module 
import features as feat 
import reranking as rank 
import eval as eval_module
from create_groundtruth import create_groundtruth_from_query
import os
import polars as pl
import pandas as pd
import pickle
import json
from tqdm import tqdm
import shutil
import gc as python_gc
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- GLOBAL DECLARATION ---
root_dir = 'data/'
cache_tag = 'v14a_repurchase_due_only'
candidate_cache_tag = 'v14a_repurchase_due_only'

try:
    if os.path.exists(root_dir):
        item_lf = pl.scan_parquet(os.path.join(root_dir, 'items.parquet'))
        user_lf = None  # Not available in current dataset
        transaction_lf = pl.scan_parquet(os.path.join(root_dir, 'transaction_full_2025.parquet'))
        event_path_parquet = os.path.join(root_dir, 'event_full_2025.parquet')
        event_path_csv = os.path.join(root_dir, 'event_full_2025.csv')
        if os.path.exists(event_path_parquet):
            event_lf = pl.scan_parquet(event_path_parquet)
        elif os.path.exists(event_path_csv):
            event_lf = pl.scan_csv(event_path_csv)
        else:
            event_lf = None
    else:
        transaction_lf = None; item_lf = None; event_lf = None
except Exception as e:
    print(f"âš ï¸ Warning: Error loading data files: {e}")
    transaction_lf = None; item_lf = None; event_lf = None


def _standardize_source_lf(source_lf, use_event_weights=False):
    if source_lf is None:
        return None

    sample_df = source_lf.select(pl.col("*")).limit(1).collect()
    available_cols = sample_df.columns
    available_set = set(available_cols)

    def _datetime_expr(col_name):
        sample_val = sample_df.select(col_name)[0, 0]
        expr = pl.col(col_name)
        if isinstance(sample_val, str):
            return pl.coalesce([
                expr.str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                expr.str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                expr.str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime),
                expr.str.to_datetime(strict=False),
            ])
        return expr.cast(pl.Datetime, strict=False)

    if "updated_date" in available_cols:
        date_expr = _datetime_expr("updated_date")
    elif "created_date" in available_cols:
        date_expr = _datetime_expr("created_date")
    elif "event_date" in available_cols:
        date_expr = _datetime_expr("event_date")
    elif {"created_year", "created_month", "created_day"}.issubset(available_set):
        date_expr = pl.date("created_year", "created_month", "created_day").cast(pl.Datetime)
    else:
        raise ValueError(f"Cannot find date column. Available: {available_cols}")

    quantity_base = pl.col("quantity").fill_null(1) if "quantity" in available_cols else pl.lit(1)

    if use_event_weights and "event_type" in available_cols:
        event_type_norm = (
            pl.col("event_type")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
        )
        quantity_expr = (
            pl.when(event_type_norm == "view_item").then(pl.lit(1))
            .when(event_type_norm == "add_to_cart").then(pl.lit(3))
            .when(event_type_norm == "purchase").then(pl.lit(5))
            .otherwise(quantity_base)
            .cast(pl.Int64)
            .alias("quantity")
        )

        return (
            source_lf
            .with_columns([
                date_expr.cast(pl.Datetime).alias("created_date"),
                event_type_norm.alias("event_type"),
                quantity_expr
            ])
            .select([
                pl.col("customer_id"),
                pl.col("item_id"),
                pl.col("quantity").cast(pl.Int64),
                pl.col("created_date").cast(pl.Datetime),
                pl.col("event_type")
            ])
        )

    return (
        source_lf
        .with_columns([
            date_expr.cast(pl.Datetime).alias("created_date"),
            pl.lit(5, dtype=pl.Int64).alias("quantity"),
            pl.lit("purchase").alias("event_type")
        ])
        .select([
            pl.col("customer_id"),
            pl.col("item_id"),
            pl.col("quantity"),
            pl.col("created_date").cast(pl.Datetime),
            pl.col("event_type")
        ])
    )


def _validate_data_quality(df_name, lf):
    """Comprehensive data quality checks to catch issues early."""
    if lf is None:
        return True
    
    print(f"\nðŸ“Š Data Quality Check: {df_name}")
    df_sample = lf.limit(100000).collect()
    
    # Check for string placeholders in numeric columns
    for col in ["customer_id", "item_id"]:
        if col in df_sample.columns:
            col_dtype = df_sample[col].dtype
            if col_dtype == pl.Utf8 or col_dtype == pl.String:
                invalid_vals = df_sample.filter(
                    pl.col(col).is_in(["(not set)", "not set", "", "None", "NaN"])
                ).height
                if invalid_vals > 0:
                    print(f"   WARNING: Found {invalid_vals} invalid values in '{col}'")
    
    print(f"   OK: Data quality check passed")
    return True


CANDIDATE_FEATURE_DEFAULTS = {
    "candidate_score": 0.0,
    "candidate_rank": 9999,
    "source_purchase_cf": 0,
    "source_atc_cf": 0,
    "source_view_cf": 0,
    "source_trending": 0,
    "source_category_trending": 0,
    "source_repeat_purchase": 0,
    "source_repurchase_due": 0,
    "source_recent_velocity_30d": 0,
    "source_high_conversion": 0,
    "repurchase_due_source_score": 0.0,
    "recent_velocity_source_score": 0.0,
    "conversion_source_score": 0.0,
    "num_candidate_sources": 0,
}


def _attach_candidate_features(label_df, candidate_df):
    candidate_cols = [c for c in CANDIDATE_FEATURE_DEFAULTS if c in candidate_df.columns]
    if not candidate_cols:
        return label_df

    candidate_features = (
        candidate_df
        .select(["customer_id", "item_id"] + candidate_cols)
        .with_columns([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id_join"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id_join"),
        ])
        .drop(["customer_id", "item_id"])
        .unique(subset=["customer_id_join", "item_id_join"])
    )

    return (
        label_df
        .with_columns([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id_join"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id_join"),
        ])
        .join(candidate_features, on=["customer_id_join", "item_id_join"], how="left")
        .drop(["customer_id_join", "item_id_join"])
    )


def _candidate_recall_report(candidates_df, pos_df, label, ks=(80, 150, 200, 300)):
    cand = candidates_df.select([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ])
    pos = pos_df.select([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).unique()

    total_pos = pos.height
    total_users = pos.select("customer_id").unique().height
    print(f"\n   Candidate Recall Diagnostics ({label})")
    print(f"   Positive pairs: {total_pos}, GT users: {total_users}")

    for k in list(ks) + [None]:
        if k is None:
            cand_k = cand
            suffix = "ALL"
        else:
            cand_k = (
                cand
                .with_row_count("_candidate_order")
                .sort(["customer_id", "_candidate_order"])
                .group_by("customer_id", maintain_order=True)
                .head(k)
                .drop("_candidate_order")
            )
            suffix = str(k)

        hits = pos.join(cand_k, on=["customer_id", "item_id"], how="inner")
        hit_users = hits.select("customer_id").unique().height
        recall = hits.height / total_pos if total_pos else 0.0
        user_hit_rate = hit_users / total_users if total_users else 0.0
        print(f"   Recall@{suffix}: {recall:.6f} | UserHit@{suffix}: {user_hit_rate:.6f} | hits={hits.height}")


def _print_recommendation_count_stats(df, label):
    if df.height == 0:
        print(f"{label}: rows=0, customers=0")
        return {"rows": 0, "customers": 0, "min": 0, "max": 0, "mean": 0.0, "short": 0}

    counts = df.group_by("customer_id").len()
    stats = counts.select([
        pl.col("len").min().alias("min"),
        pl.col("len").max().alias("max"),
        pl.col("len").mean().alias("mean"),
        (pl.col("len") < 10).sum().alias("short"),
    ]).row(0, named=True)
    print(f"{label}:")
    print(f"   rows: {df.height}")
    print(f"   customers: {counts.height}")
    print(f"   recs/customer min/max/mean: {stats['min']}/{stats['max']}/{stats['mean']:.2f}")
    print(f"   users with fewer than 10 recs: {stats['short']}")
    return {"rows": df.height, "customers": counts.height, **stats}


def _prepare_submission_frame(df_final):
    required = {"customer_id", "item_id", "pred_score"}
    missing = required - set(df_final.columns)
    if missing:
        raise ValueError(f"Missing required final submission columns: {missing}")

    df = (
        df_final
        .select([
            pl.col("customer_id").cast(pl.Utf8).str.strip_chars().alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("pred_score").fill_null(float("-inf")).alias("pred_score"),
        ] + [
            pl.col(c) for c in ["candidate_score", "candidate_rank"] if c in df_final.columns
        ])
        .filter(
            pl.col("customer_id").is_not_null()
            & (pl.col("customer_id") != "")
            & pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
    )

    sort_cols = ["customer_id", "pred_score"]
    descending = [False, True]
    if "candidate_score" in df.columns:
        sort_cols.append("candidate_score")
        descending.append(True)
    if "candidate_rank" in df.columns:
        sort_cols.append("candidate_rank")
        descending.append(False)
    sort_cols.append("item_id")
    descending.append(False)

    return (
        df
        .sort(sort_cols, descending=descending)
        .unique(subset=["customer_id", "item_id"], keep="first", maintain_order=True)
        .group_by("customer_id", maintain_order=True)
        .head(10)
        .select(["customer_id", "item_id", "pred_score"])
    )


def _build_result_dict(df_submit):
    grouped = (
        df_submit
        .sort(["customer_id", "pred_score", "item_id"], descending=[False, True, False])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").alias("items"))
    )
    return {
        int(str(cust_id).strip()): [str(item).strip() for item in list(items)[:10]]
        for cust_id, items in grouped.iter_rows()
    }


def _get_safe_target_customers(
    run_mode,
    user_lf,
    df_candidates,
    purchase_lf_processed,
    queries,
    target_customers=None,
):
    """Return a leakage-safe fallback target universe with one customer_id column."""
    if target_customers is not None:
        return pl.DataFrame({"customer_id": list(target_customers)}).select("customer_id").unique(), "target_customers"

    if user_lf is not None:
        return user_lf.select("customer_id").unique(), "user_lf"

    # Do not use full purchase_lf_processed here: it includes target-window
    # purchases in local validation/test modes and would leak future user IDs.
    return df_candidates.select("customer_id").unique(), "df_candidates"


def main():
    try:
        print(">>> START PIPELINE (High Precision Strategy: 5-6%) >>>")
        
        cfg = load_config("params.json")
        run_mode = os.environ.get("RUN_MODE", "local_test").strip().lower()
        if run_mode not in {"val", "local_test", "private"}:
            raise ValueError("RUN_MODE must be one of: val, local_test, private")
        print(f">>> RUN_MODE: {run_mode}")
        final_path = "final_submission.parquet" 
        model_path = f"lgbm_model_{cache_tag}.pkl" 
        
        df_final = None

        if os.path.exists(final_path):
            print(f"   Existing {final_path} will be rebuilt for cache_tag={cache_tag}.")
            print(f"\nâœ… FOUND EXISTING RESULT: {final_path}")
            # df_final = pl.read_parquet(final_path) # Uncomment náº¿u muá»‘n resume tá»« file káº¿t quáº£
        
        if df_final is None:
            if transaction_lf is None: raise ValueError("Data error.")
            
            print(">> Processing data...")
            purchase_lf_processed = _standardize_source_lf(transaction_lf, use_event_weights=False).drop_nulls(subset=["customer_id"])
            event_lf_processed = _standardize_source_lf(event_lf, use_event_weights=True)
            if event_lf_processed is not None:
                event_lf_processed = event_lf_processed.drop_nulls(subset=["customer_id"])
                event_lf_processed = event_lf_processed.filter(pl.col("event_type").is_in(["view_item", "add_to_cart"]))
                interaction_lf_processed = pl.concat([purchase_lf_processed, event_lf_processed], how="vertical_relaxed")
            else:
                interaction_lf_processed = purchase_lf_processed
            
            queries = cfg.create_query_string()
            
            stage1_cfg = {
                "top_n": 200,
                "n_trend": 200,
                "neighbors_per_model": 20,
                "recent_purchase_n": 10,
                "recent_atc_n": 10,
                "recent_view_n": 20,
                "purchase_seed_weight": 5.0,
                "atc_seed_weight": 3.0,
                "view_seed_weight": 1.0,
                "allow_repeat_consumables": True,
                "enable_category_trending": True,
                "category_source_weight": 2.5,
                "max_user_categories": 3,
                "max_category_items": 50,
                "enable_repeat_candidates": True,
                "repeat_source_weight": 4.0,
                "max_repeat_items_per_user": 20,
                "enable_recent_trending": True,
                "recent_trending_days": 60,
                "recent_trending_weight": 1.5,
                "enable_repurchase_due_candidates": True,
                "repurchase_due_weight": 3.5,
                "max_repurchase_due_items_per_user": 20,
                "repurchase_due_min_days": 7,
                "repurchase_due_max_days": 120,
                "enable_recent_velocity_candidates": False,
                "recent_velocity_weight": 1.25,
                "max_recent_velocity_items": 200,
                "enable_high_conversion_candidates": False,
                "high_conversion_weight": 0.75,
                "max_high_conversion_items": 200,
                "min_conversion_purchase_count": 5,
            }
            stage2_cfg = {
                "n_neg_per_pos": 5,
                "min_neg_per_user": 5,
                "max_neg_per_user": 50,
                "top_n": stage1_cfg["top_n"],
                "alpha": 0.001,
                "model_name": "LightGBM",
                "session_window": 1,
                "min_coo": 1,
                "n_estimators": 2000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "train_cap": 2_000_000,
            }

            # --- STAGE 1: CANDIDATES ---
            print("\n--- STAGE 1: CANDIDATE GENERATION ---")
            save_path_s1 = f"candidates_inference_{candidate_cache_tag}.parquet"
            if os.path.exists(save_path_s1):
                print(f"âœ… Found existing candidates: {save_path_s1}")
                df_candidates = pl.read_parquet(save_path_s1)
                print(f"   Loaded {df_candidates.height} candidate records")
            else:
                print(f">> Generating inference candidates (Top {stage1_cfg['top_n']})...")
                # Final inference mode: use all observed data before the simulated test period.
                df_candidates = gc_module.get_candidates(
                    interaction_lf_processed, 
                    q_hist=queries['inference_feature_history'], 
                    top_n=stage1_cfg['top_n'],
                    q_val=None,
                    event_lf=event_lf_processed,
                    item_lf=item_lf,
                    include_val_for_trending=False,
                    n_trend=stage1_cfg["n_trend"],
                    neighbors_per_model=stage1_cfg["neighbors_per_model"],
                    recent_purchase_n=stage1_cfg["recent_purchase_n"],
                    recent_atc_n=stage1_cfg["recent_atc_n"],
                    recent_view_n=stage1_cfg["recent_view_n"],
                    purchase_seed_weight=stage1_cfg["purchase_seed_weight"],
                    atc_seed_weight=stage1_cfg["atc_seed_weight"],
                    view_seed_weight=stage1_cfg["view_seed_weight"],
                    allow_repeat_consumables=stage1_cfg["allow_repeat_consumables"],
                    enable_category_trending=stage1_cfg["enable_category_trending"],
                    category_source_weight=stage1_cfg["category_source_weight"],
                    max_user_categories=stage1_cfg["max_user_categories"],
                    max_category_items=stage1_cfg["max_category_items"],
                    enable_repeat_candidates=stage1_cfg["enable_repeat_candidates"],
                    repeat_source_weight=stage1_cfg["repeat_source_weight"],
                    max_repeat_items_per_user=stage1_cfg["max_repeat_items_per_user"],
                    enable_recent_trending=stage1_cfg["enable_recent_trending"],
                    recent_trending_days=stage1_cfg["recent_trending_days"],
                    recent_trending_weight=stage1_cfg["recent_trending_weight"],
                    enable_repurchase_due_candidates=stage1_cfg["enable_repurchase_due_candidates"],
                    repurchase_due_weight=stage1_cfg["repurchase_due_weight"],
                    max_repurchase_due_items_per_user=stage1_cfg["max_repurchase_due_items_per_user"],
                    repurchase_due_min_days=stage1_cfg["repurchase_due_min_days"],
                    repurchase_due_max_days=stage1_cfg["repurchase_due_max_days"],
                    enable_recent_velocity_candidates=stage1_cfg["enable_recent_velocity_candidates"],
                    recent_velocity_weight=stage1_cfg["recent_velocity_weight"],
                    max_recent_velocity_items=stage1_cfg["max_recent_velocity_items"],
                    enable_high_conversion_candidates=stage1_cfg["enable_high_conversion_candidates"],
                    high_conversion_weight=stage1_cfg["high_conversion_weight"],
                    max_high_conversion_items=stage1_cfg["max_high_conversion_items"],
                    min_conversion_purchase_count=stage1_cfg["min_conversion_purchase_count"],
                )
                print(f"âœ… Generated {df_candidates.height} candidate records")
                df_candidates.write_parquet(save_path_s1)
                print(f"âœ… Saved to {save_path_s1}")

            # --- STAGE 2: TRAINING ---
            print("\n--- STAGE 2: TRAINING MODEL ---")  
            feature_cols = [
                "item_pop_log",
                "cooc_max",
                "cooc_mean",
                "cooc_len",
                "days_since_last_brand_purchase",
                "brand_cross_score",
                "price_ratio",
                "days_since_last_cat3_purchase",
                "user_recency_days",
                "user_frequency_log",
                "user_monetary_log",
                "user_rfm_score",
                "user_brand_purchase_share",
                "user_cat3_purchase_share",
                "brand_recency_x_freq",
                "cat3_recency_x_freq",
                "user_stickiness_score",
                "user_repeat_cat3_ratio",
                "item_repeat_rate",
                "cat3_repeat_rate",
                "days_since_last_item_purchase",
                "user_item_purchase_count",
                "user_item_avg_repurchase_days",
                "user_cat3_avg_repurchase_days",
                "item_repurchase_due_score",
                "cat3_repurchase_due_score",
                "item_pop_30d",
                "item_pop_prev30d",
                "item_velocity_30d",
                "cat3_pop_30d",
                "cat3_pop_prev30d",
                "cat3_velocity_30d",
                "item_view_to_purchase_rate",
                "item_atc_to_purchase_rate",
                "cat3_view_to_purchase_rate",
                "cat3_atc_to_purchase_rate",
                "user_view_count_log",
                "user_atc_count_log",
                "user_view_unique_items_log",
                "user_atc_unique_items_log",
                "days_since_last_view",
                "days_since_last_atc",
                "user_view_atc_ratio",
                "view_item_match",
                "atc_match",
                "user_item_view_count",
                "user_item_atc_count",
                "candidate_score",
                "candidate_rank",
                "source_purchase_cf",
                "source_atc_cf",
                "source_view_cf",
                "source_trending",
                "source_category_trending",
                "source_repeat_purchase",
                "source_repurchase_due",
                "repurchase_due_source_score",
                "num_candidate_sources"
            ]
            PRUNED_FEATURES = {
                "brand_match",
                "cat_l3_match",
                "user_brand_count",
                "user_cat3_freq_log",
            }
            assert not any(f in feature_cols for f in PRUNED_FEATURES)

            def load_or_generate_candidates(split_name, q_hist, target_customers=None):
                save_path = f"candidates_{split_name}_{candidate_cache_tag}.parquet"
                if os.path.exists(save_path):
                    print(f"Found existing {split_name} candidates: {save_path}")
                    df = pl.read_parquet(save_path)
                    print(f"   Loaded {df.height} candidate records")
                    return df

                print(f">> Generating {split_name} candidates (Top {stage1_cfg['top_n']})...")
                df = gc_module.get_candidates(
                    interaction_lf_processed,
                    q_hist=q_hist,
                    top_n=stage1_cfg["top_n"],
                    q_val=None,
                    event_lf=event_lf_processed,
                    item_lf=item_lf,
                    target_customers=target_customers,
                    include_val_for_trending=False,
                    n_trend=stage1_cfg["n_trend"],
                    neighbors_per_model=stage1_cfg["neighbors_per_model"],
                    recent_purchase_n=stage1_cfg["recent_purchase_n"],
                    recent_atc_n=stage1_cfg["recent_atc_n"],
                    recent_view_n=stage1_cfg["recent_view_n"],
                    purchase_seed_weight=stage1_cfg["purchase_seed_weight"],
                    atc_seed_weight=stage1_cfg["atc_seed_weight"],
                    view_seed_weight=stage1_cfg["view_seed_weight"],
                    allow_repeat_consumables=stage1_cfg["allow_repeat_consumables"],
                    enable_category_trending=stage1_cfg["enable_category_trending"],
                    category_source_weight=stage1_cfg["category_source_weight"],
                    max_user_categories=stage1_cfg["max_user_categories"],
                    max_category_items=stage1_cfg["max_category_items"],
                    enable_repeat_candidates=stage1_cfg["enable_repeat_candidates"],
                    repeat_source_weight=stage1_cfg["repeat_source_weight"],
                    max_repeat_items_per_user=stage1_cfg["max_repeat_items_per_user"],
                    enable_recent_trending=stage1_cfg["enable_recent_trending"],
                    recent_trending_days=stage1_cfg["recent_trending_days"],
                    recent_trending_weight=stage1_cfg["recent_trending_weight"],
                    enable_repurchase_due_candidates=stage1_cfg["enable_repurchase_due_candidates"],
                    repurchase_due_weight=stage1_cfg["repurchase_due_weight"],
                    max_repurchase_due_items_per_user=stage1_cfg["max_repurchase_due_items_per_user"],
                    repurchase_due_min_days=stage1_cfg["repurchase_due_min_days"],
                    repurchase_due_max_days=stage1_cfg["repurchase_due_max_days"],
                    enable_recent_velocity_candidates=stage1_cfg["enable_recent_velocity_candidates"],
                    recent_velocity_weight=stage1_cfg["recent_velocity_weight"],
                    max_recent_velocity_items=stage1_cfg["max_recent_velocity_items"],
                    enable_high_conversion_candidates=stage1_cfg["enable_high_conversion_candidates"],
                    high_conversion_weight=stage1_cfg["high_conversion_weight"],
                    max_high_conversion_items=stage1_cfg["max_high_conversion_items"],
                    min_conversion_purchase_count=stage1_cfg["min_conversion_purchase_count"],
                )
                print(f"Generated {df.height} {split_name} candidate records")
                df.write_parquet(save_path)
                print(f"Saved to {save_path}")
                return df

            if os.path.exists(model_path):
                print(f"âœ… Found existing model: {model_path}")
                with open(model_path, 'rb') as f: model = pickle.load(f)
            else:
                print(">> Starting Training...")
                print("\n   [Phase 1/5] Positive Sampling (Train/Val)...")
                pos_train_df, train_cust_ids = rank.positive_sampling(purchase_lf_processed, queries['train_target'])
                pos_val_df, val_cust_ids = rank.positive_sampling(purchase_lf_processed, queries['val_target'])
                print(f"   âœ… Train customers: {len(train_cust_ids)}, train positives: {pos_train_df.height}")
                print(f"   âœ… Val customers:   {len(val_cust_ids)}, val positives:   {pos_val_df.height}")
                
                print("\n   [Phase 2/5] Split Candidate Generation + Negative Sampling...")
                # Local validation mode: each candidate pool only sees feature history before its target window.
                train_candidates = load_or_generate_candidates(
                    "train",
                    queries["train_feature_history"],
                    target_customers=train_cust_ids,
                )
                val_candidates = load_or_generate_candidates(
                    "val",
                    queries["val_feature_history"],
                    target_customers=val_cust_ids,
                )
                _candidate_recall_report(train_candidates, pos_train_df, "train")
                _candidate_recall_report(val_candidates, pos_val_df, "val")
                neg_train_df = rank.negative_sampling(train_candidates, pos_train_df, stage2_cfg)
                neg_val_df = rank.negative_sampling(val_candidates, pos_val_df, stage2_cfg)
                print(f"   âœ… Train negatives: {neg_train_df.height}")
                print(f"   âœ… Val negatives:   {neg_val_df.height}")
                
                print("\n   [Phase 3/5] Feature Engineering (Train/Val)...")
                train_target_user_type = pos_train_df.schema["customer_id"]
                train_target_item_type = pos_train_df.schema["item_id"]
                train_target_target_type = pos_train_df.schema["target"]
                val_target_user_type = pos_val_df.schema["customer_id"]
                val_target_item_type = pos_val_df.schema["item_id"]
                val_target_target_type = pos_val_df.schema["target"]

                neg_train_df = neg_train_df.select([
                    pl.col("customer_id").cast(train_target_user_type),
                    pl.col("item_id").cast(train_target_item_type),
                    pl.col("target").cast(train_target_target_type)
                ])
                neg_val_df = neg_val_df.select([
                    pl.col("customer_id").cast(val_target_user_type),
                    pl.col("item_id").cast(val_target_item_type),
                    pl.col("target").cast(val_target_target_type)
                ])

                df_train_raw = _attach_candidate_features(pl.concat([pos_train_df, neg_train_df]), train_candidates)
                df_val_raw = _attach_candidate_features(pl.concat([pos_val_df, neg_val_df]), val_candidates)

                train_feature_dir = f"temp_features_train_{cache_tag}"
                val_feature_dir = f"temp_features_val_{cache_tag}"
                if os.path.exists(train_feature_dir): shutil.rmtree(train_feature_dir)
                if os.path.exists(val_feature_dir): shutil.rmtree(val_feature_dir)

                df_train_features_lazy = feat.generate_features(
                    candidates_df=df_train_raw,
                    transaction_lf=purchase_lf_processed,
                    item_lf=item_lf,
                    event_lf=event_lf_processed,
                    queries=queries,
                    cfg=cfg,
                    feature_cols=feature_cols,
                    mode_name=f"train_{cache_tag}",
                    feature_history_query=queries["train_feature_history"],
                )
                df_val_features_lazy = feat.generate_features(
                    candidates_df=df_val_raw,
                    transaction_lf=purchase_lf_processed,
                    item_lf=item_lf,
                    event_lf=event_lf_processed,
                    queries=queries,
                    cfg=cfg,
                    feature_cols=feature_cols,
                    mode_name=f"val_{cache_tag}",
                    feature_history_query=queries["val_feature_history"],
                )

                print("\n   [Phase 4/5] Collecting Training Data...")
                df_train_features = df_train_features_lazy.collect()
                print(f"   âœ… Loaded {df_train_features.height} training samples")

                print("   >> Collecting Validation Data...")
                df_val_features = df_val_features_lazy.collect()
                print(f"   âœ… Loaded {df_val_features.height} validation samples")

                train_cap = stage2_cfg["train_cap"]
                if df_train_features.height > train_cap:
                    print(f"   >> Sampling down to {train_cap:,}...")
                    df_train_features = df_train_features.sample(n=train_cap, seed=42)
                    print(f"   âœ… Sampled to {df_train_features.height} training samples")

                print("\n   [Phase 5/5] Model Training...")
                model = rank.train_model(
                    df_train_features,
                    feature_cols,
                    stage2_cfg["model_name"],
                    stage2_cfg,
                    df_val=df_val_features,
                    early_stopping_rounds=50
                )
                print(f"   âœ… Model training completed")
                
                print("\n   [Validation Report] Ranking on full validation candidate pool...")
                val_pred_mode = f"val_inference_{cache_tag}"
                val_pred_dir = f"temp_features_{val_pred_mode}"
                if os.path.exists(val_pred_dir): shutil.rmtree(val_pred_dir)
                _ = feat.generate_features(
                    candidates_df=val_candidates,
                    transaction_lf=purchase_lf_processed,
                    item_lf=item_lf,
                    event_lf=event_lf_processed,
                    queries=queries,
                    cfg=cfg,
                    model=model,
                    feature_cols=feature_cols,
                    mode_name=val_pred_mode,
                    feature_history_query=queries["val_feature_history"],
                )
                try:
                    df_val_pred = pl.scan_parquet(f"{val_pred_dir}/part_*.parquet").collect()
                    eval_module.evaluate_predictions_df(df_val_pred, pos_val_df, k=10)
                    del df_val_pred
                except Exception as e:
                    print(f"   Warning: validation ranking report failed: {e}")

                # if hasattr(model, "feature_importances_"):
                #     print("\nðŸ“Š LIGHTGBM IMPORTANCE:")
                #     try:
                #         imp = pd.DataFrame({"Feature": feature_cols, "Value": model.feature_importances_})
                #         print(imp.sort_values(by="Value", ascending=False).to_string(index=False))
                #     except: pass
                #     print("="*40)

                with open(model_path, 'wb') as f: pickle.dump(model, f)
                del df_train_features, df_val_features, df_train_raw, df_val_raw
                python_gc.collect()
                if os.path.exists(train_feature_dir): shutil.rmtree(train_feature_dir)
                if os.path.exists(val_feature_dir): shutil.rmtree(val_feature_dir)

            # --- STAGE 3: INFERENCE & HYBRID RERANKING ---
            print("\n--- STAGE 3: INFERENCE (HYBRID RERANKING) ---")
            print(">> [Phase 1] Generating predictions on candidates...")

            # # Speed-up private inference: only rerank top 100 Stage-1 candidates/user
            # df_candidates_infer = df_candidates.filter(pl.col("candidate_rank") <= 100)
            # print(f"   Using top-100 inference candidates: {df_candidates_infer.height} / {df_candidates.height}")

            _ = feat.generate_features(
                candidates_df=df_candidates,
                transaction_lf=purchase_lf_processed,
                item_lf=item_lf,
                event_lf=event_lf_processed,
                queries=queries,
                cfg=cfg,
                model=model,
                feature_cols=feature_cols,
                mode_name=f"inference_{cache_tag}",
                feature_history_query=queries["inference_feature_history"],
            )
            print("âœ… Predictions generated")
            
            print(">> [Phase 2] Model-first top-10 selection...")
            inference_dir = f"temp_features_inference_{cache_tag}"
            top10_dir = "temp_top10_chunks"
            if os.path.exists(top10_dir): shutil.rmtree(top10_dir)
            os.makedirs(top10_dir)
            
            pred_files = [os.path.join(inference_dir, f) for f in os.listdir(inference_dir) if f.endswith('.parquet') and f != "_SUCCESS"]
            
            prediction_rows_before_top10 = 0
            prediction_customers = set()
            score_min = None
            score_max = None
            score_sum = 0.0
            score_count = 0

            for i, fpath in enumerate(tqdm(pred_files, desc="Model-first Ranking")):
                try:
                    df_chunk = pl.read_parquet(fpath)
                    if not {"customer_id", "item_id", "pred_score"}.issubset(set(df_chunk.columns)):
                        print(f"Warning: missing required prediction columns in {fpath}")
                        continue

                    prediction_rows_before_top10 += df_chunk.height
                    prediction_customers.update(df_chunk.select("customer_id").unique()["customer_id"].to_list())
                    score_stats = df_chunk.select([
                        pl.col("pred_score").min().alias("min"),
                        pl.col("pred_score").max().alias("max"),
                        pl.col("pred_score").sum().alias("sum"),
                        pl.col("pred_score").count().alias("count"),
                    ]).row(0, named=True)
                    if score_stats["count"]:
                        score_min = score_stats["min"] if score_min is None else min(score_min, score_stats["min"])
                        score_max = score_stats["max"] if score_max is None else max(score_max, score_stats["max"])
                        score_sum += float(score_stats["sum"] or 0.0)
                        score_count += int(score_stats["count"])

                    df_chunk = df_chunk.with_columns(pl.col("pred_score").alias("final_score"))
                    sort_cols = ["customer_id", "final_score"]
                    descending = [False, True]
                    for col, desc in [
                        ("candidate_score", True),
                        ("candidate_rank", False),
                        ("atc_match", True),
                        ("view_item_match", True),
                        ("item_pop_log", True),
                        ("item_id", False),
                    ]:
                        if col in df_chunk.columns:
                            sort_cols.append(col)
                            descending.append(desc)

                    df_chunk_top = (
                        df_chunk
                        .sort(sort_cols, descending=descending)
                        .group_by("customer_id", maintain_order=True)
                        .head(10)
                        .select(["customer_id", "item_id", pl.col("final_score").alias("pred_score")])
                    )

                    if df_chunk_top.height > 0:
                        df_chunk_top.write_parquet(f"{top10_dir}/top10_part_{i}.parquet")
                    del df_chunk, df_chunk_top
                except Exception as e:
                    print(f"Warning: error reading {fpath}: {e}")

            print(">> Final Merge...")
            try:
                df_final = (
                    pl.scan_parquet(f"{top10_dir}/*.parquet")
                    .sort(["customer_id", "pred_score", "item_id"], descending=[False, True, False])
                    .group_by("customer_id", maintain_order=True)
                    .head(10)
                    .collect()
                )
                print(f"Merged {df_final.height} records")
            except Exception:
                print("No predictions found, creating empty dataframe")
                df_final = pl.DataFrame({"customer_id": [], "item_id": [], "pred_score": []}, schema={"customer_id": pl.Int64, "item_id": pl.String, "pred_score": pl.Float32})

            if os.path.exists(top10_dir): shutil.rmtree(top10_dir)

            avg_candidates = prediction_rows_before_top10 / len(prediction_customers) if prediction_customers else 0.0
            score_mean = score_sum / score_count if score_count else 0.0
            print(">> Stage 3 Diagnostics:")
            print(f"   Prediction rows before top-10: {prediction_rows_before_top10}")
            print(f"   Customers with predictions: {len(prediction_customers)}")
            print(f"   Avg candidates/customer before top-10: {avg_candidates:.2f}")
            print(f"   Rows after top-10: {df_final.height}")
            print(f"   Pred score min/max/mean: {score_min}/{score_max}/{score_mean:.6f}")
            # --- STAGE 3.5: SAFE FALLBACK ---
            print("\n--- STAGE 3.5: FALLBACK (Global Trending) ---")

            all_custs, target_source = _get_safe_target_customers(
                run_mode=run_mode,
                user_lf=user_lf,
                df_candidates=df_candidates,
                purchase_lf_processed=purchase_lf_processed,
                queries=queries,
                target_customers=None,
            )
            all_custs_df = all_custs.collect() if isinstance(all_custs, pl.LazyFrame) else all_custs
            if df_final.height > 0 and all_custs_df.height > 0:
                all_custs_df = all_custs_df.with_columns(
                    pl.col("customer_id").cast(df_final.schema["customer_id"], strict=False)
                )
            safe_target_count = all_custs_df.height

            user_counts = df_final.group_by("customer_id").len()
            
            users_short = (
                all_custs_df
                .join(user_counts, on="customer_id", how="left")
                .filter(pl.col("len").fill_null(0) < 10)
                .select("customer_id")
            )
            
            missing_custs_list = users_short["customer_id"].to_list()
            print(f">> Checking coverage...")
            print(f"   Safe fallback target source: {target_source}")
            print(f"   Safe fallback target customers: {safe_target_count}")
            print(f"   With recommendations before fallback: {df_final.select('customer_id').unique().height}")
            print(f"   Needing fallback: {len(missing_custs_list)}")

            if len(missing_custs_list) > 0:
                print(f">> Filling with Top 10 Global Trending...")
                global_pop = gc_module.get_trending_items(
                    interaction_lf_processed,
                    queries['inference_feature_history'],
                    None,
                    n_trend=50,
                    include_val_for_trending=False,
                )
                n_pop = len(global_pop)
                
                batch_size = 50000
                fallback_dfs = []
                target_schema = df_final.schema 
                
                print(f"   Processing {len(missing_custs_list)} customers in batches...")
                for i in tqdm(range(0, len(missing_custs_list), batch_size), desc="Fallback"):
                    chunk_u = missing_custs_list[i : i + batch_size]
                    
                    chunk_df = pl.DataFrame({
                        "customer_id": np.repeat(chunk_u, n_pop),
                        "item_id": np.tile(global_pop, len(chunk_u)),
                        "pred_score": np.full(len(chunk_u)*n_pop, -1.0e9, dtype=np.float32)
                    })
                    
                    chunk_df = chunk_df.select([
                        pl.col("customer_id").cast(target_schema["customer_id"]),
                        pl.col("item_id").cast(target_schema["item_id"]),
                        pl.col("pred_score").cast(target_schema["pred_score"])
                    ])
                    
                    fallback_dfs.append(chunk_df)
                
                if fallback_dfs:
                    print("   >> Merging fallback...")
                    full_fallback = pl.concat(fallback_dfs)
                    df_final = (
                        pl.concat([df_final, full_fallback])
                        .unique(subset=["customer_id", "item_id"], keep="first", maintain_order=True)
                    )
                    
                    df_final = (
                        df_final
                        .sort(["customer_id", "pred_score", "item_id"], descending=[False, True, False])
                        .group_by("customer_id", maintain_order=True)
                        .head(10)
                    )
                    print(f"âœ… Fallback completed")

            final_user_count = df_final.select("customer_id").unique().height
            final_avg_recs = df_final.height / final_user_count if final_user_count else 0.0
            print(f"   Customers after fallback: {final_user_count}")
            print(f"   Avg recommendations/customer after fallback: {final_avg_recs:.2f}")
            if final_user_count > safe_target_count:
                raise ValueError(
                    f"Fallback customer count exceeded safe target universe: "
                    f"{final_user_count} > {safe_target_count}"
                )

            df_final.write_parquet(final_path)
            print(f"âœ… Saved final submission to: {final_path}")

        # --- STAGE 4: EXPORT & EVALUATION ---
        print("\n--- STAGE 4: EXPORT & EVALUATION ---")
        json_path = "result_v11.json"
        pkl_path = "result_v11.pkl"
        eval_report_path = "evaluation_report_v11.json"
        export_report_path = "export_report_v11.json"
        
        print(">> [Phase 1] Preparing submit-safe recommendations...")
        _print_recommendation_count_stats(df_final, "Before export preparation")
        df_submit = _prepare_submission_frame(df_final)
        export_stats = _print_recommendation_count_stats(df_submit, "After export preparation")
        df_submit.write_parquet(final_path)
        print(f"Submit-safe final submission saved to: {final_path}")
        
        print(f">> [Phase 2] Exporting to {json_path}...")
        try:
            result_dict = _build_result_dict(df_submit)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False)
            print("Exported JSON successfully.")
        except Exception as e:
            print(f"Error writing JSON: {e}")
            result_dict = {}
        
        # Also export as pickle dict
        print(f">> [Phase 3] Saving results to {pkl_path}...")
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(result_dict, f)
            print("Exported pickle dictionary successfully.")
        except Exception as e:
            print(f"Error writing pickle: {e}")

        export_report = {
            "run_mode": run_mode,
            "cache_tag": cache_tag,
            "candidate_cache_tag": candidate_cache_tag,
            "n_customers_exported": export_stats["customers"],
            "n_rows_exported": export_stats["rows"],
        }
        with open(export_report_path, "w", encoding="utf-8") as f:
            json.dump(export_report, f, indent=2)

        print(f">> [Phase 4] Evaluation mode: {run_mode}")
        if run_mode == "private":
            print("Private mode: skipping evaluation because ground truth is unavailable.")
        else:
            if run_mode == "val":
                gt_path = f"data/groundtruth_val_{cache_tag}.pkl"
                gt_query = queries["val_target"]
            else:
                gt_path = f"data/groundtruth_test_dec2025_{cache_tag}.pkl"
                gt_query = queries.get("test") or queries.get("test_target")

            print(f"Ground truth path: {gt_path}")
            if not os.path.exists(gt_path):
                print("Ground truth file not found; creating split-specific ground truth...")
                create_groundtruth_from_query(
                    purchase_lf_processed,
                    gt_query,
                    gt_path,
                    date_col="created_date",
                    event_col="event_type",
                )

            print(">> Starting Evaluation...")
            eval_result = eval_module.evaluate(df_submit, k=10, gt_path=gt_path)
            eval_result.update({
                "run_mode": run_mode,
                "cache_tag": cache_tag,
                "candidate_cache_tag": candidate_cache_tag,
                "gt_path": gt_path,
            })

            print(f">> Saving evaluation report to {eval_report_path}...")
            with open(eval_report_path, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)
            print("Evaluation report saved.")
        
        # --- FINAL SUMMARY ---
        print("\n" + "="*60)
        print("ðŸŽ‰ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"ðŸ“„ Output files:")
        print(f"   â€¢ {json_path} (JSON recommendations)")
        print(f"   â€¢ {pkl_path} (Pickle backup)")
        if run_mode != "private":
            print(f"   â€¢ {eval_report_path} (Evaluation metrics)")
        print(f"   â€¢ {export_report_path} (Export metadata)")
        print(f"   â€¢ {final_path} (Raw predictions)")
        print("="*60)

    except Exception as e:
        print(f"\nâŒ ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
