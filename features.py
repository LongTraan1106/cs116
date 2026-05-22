import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import gc
import os
import shutil
import datetime

# Thử import logic tối ưu từ reranking_lasso
try:
    from reranking import get_prediction_expr
except ImportError:
    get_prediction_expr = None

warnings.filterwarnings('ignore')

CANDIDATE_FEATURE_DEFAULTS = {
    "candidate_score": 0.0,
    "candidate_rank": 9999,
    "source_purchase_cf": 0,
    "source_atc_cf": 0,
    "source_view_cf": 0,
    "source_trending": 0,
    "source_category_trending": 0,
    "source_repeat_purchase": 0,
    "num_candidate_sources": 0,
}

MARKETING_FEATURE_COLS = [
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
]

V13_FEATURE_COLS = [
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
]

V13_FEATURE_DEFAULTS = {
    "days_since_last_item_purchase": 999,
    "user_item_purchase_count": 0,
    "user_item_avg_repurchase_days": 999,
    "user_cat3_avg_repurchase_days": 999,
    "item_repurchase_due_score": 0,
    "cat3_repurchase_due_score": 0,
    "item_pop_30d": 0,
    "item_pop_prev30d": 0,
    "item_velocity_30d": 0,
    "cat3_pop_30d": 0,
    "cat3_pop_prev30d": 0,
    "cat3_velocity_30d": 0,
    "item_view_to_purchase_rate": 0,
    "item_atc_to_purchase_rate": 0,
    "cat3_view_to_purchase_rate": 0,
    "cat3_atc_to_purchase_rate": 0,
}

PRUNED_FEATURES = {
    "brand_match",
    "cat_l3_match",
    "user_brand_count",
    "user_cat3_freq_log",
}

# --- 1. PREPARE LOOKUP TABLES (FULL + TIME FEATURES) ---
def prepare_lookup_tables(transaction_lf, item_lf, q_hist, cfg, event_lf=None):
    print(">> Preparing Lookup Tables (Vectorized)...")
    item_schema = item_lf.collect_schema().names() if item_lf is not None else []
    
    # 1. Filter Data
    hist_lf = (
        transaction_lf
        .filter(pl.sql_expr(q_hist))
        .select(["customer_id", "item_id", "created_date"])
        .unique()
    )
    
    # [FEATURE 1] Global Popularity
    print("   -> Computing Global Item Popularity...")
    item_stats = hist_lf.group_by("item_id").agg(pl.len().alias("global_item_count"))
    df_item_stats = item_stats.collect()

    # 2. Lọc Spammer
    user_counts = hist_lf.group_by(["customer_id", "created_date"]).len().filter(pl.col("len") < 30)
    hist_clean = hist_lf.join(user_counts.select(["customer_id", "created_date"]), on=["customer_id", "created_date"], how="inner")
    anchor_date = hist_clean.select(pl.col("created_date").max()).collect().item()

    # [FEATURE 2] User Spending Power
    print("   -> Computing User Spending Power...")
    user_spending = None
    item_prices = None
    if item_lf is not None:
        price_col = "price" if "price" in item_schema else ("current_price" if "current_price" in item_schema else None)
        if price_col:
            item_prices = item_lf.select([pl.col("item_id"), pl.col(price_col).cast(pl.Float32).alias("price")]).unique(subset=["item_id"]).collect()
            user_spending = (
                hist_clean.join(item_prices.lazy(), on="item_id")
                .group_by("customer_id")
                .agg([pl.col("price").mean().alias("user_avg_spend")])
                .collect()
            )

    print("   -> Computing RFM purchase features...")
    user_purchase_base = (
        hist_clean
        .group_by("customer_id")
        .agg([
            pl.col("created_date").max().alias("last_purchase_date"),
            pl.len().alias("user_purchase_count"),
        ])
    )
    if item_prices is not None:
        user_monetary = (
            hist_clean
            .join(item_prices.lazy(), on="item_id", how="left")
            .group_by("customer_id")
            .agg(pl.col("price").fill_null(0).sum().alias("user_total_spend"))
        )
    else:
        user_monetary = user_purchase_base.select([
            "customer_id",
            pl.col("user_purchase_count").cast(pl.Float32).alias("user_total_spend"),
        ])

    df_user_rfm = (
        user_purchase_base
        .join(user_monetary, on="customer_id", how="left")
        .with_columns([
            (pl.lit(anchor_date) - pl.col("last_purchase_date")).dt.total_days().fill_null(999).alias("user_recency_days"),
            (pl.col("user_purchase_count").cast(pl.Float32) + 1).log().alias("user_frequency_log"),
            (pl.col("user_total_spend").fill_null(0).cast(pl.Float32) + 1).log().alias("user_monetary_log"),
        ])
        .with_columns(
            (
                (1.0 / (pl.col("user_recency_days").cast(pl.Float32) + 3).log())
                * pl.col("user_frequency_log")
                * pl.col("user_monetary_log")
            ).alias("user_rfm_score")
        )
        .select([
            "customer_id",
            "user_recency_days",
            "user_frequency_log",
            "user_monetary_log",
            "user_rfm_score",
            "user_purchase_count",
        ])
        .collect()
    )

    # [FEATURE 3 - MỚI] CATEGORY TIME CYCLES (Chu kỳ mua sắm)
    # Tính toán xem lần cuối user mua Category L3 này là bao lâu
    print("   -> Computing Category Time Cycles (RAM Safe)...")
    cat_time_stats = None
    item_cats = None
    if item_lf is not None and "category_l3" in item_schema:
        # Chỉ lấy cột cần thiết để nhẹ RAM
        item_cats = item_lf.select(["item_id", "category_l3"]).unique().collect()
        
        # Join transaction với category
        # Lưu ý: GroupBy theo (User, Cat L3) sẽ nhỏ hơn rất nhiều so với (User, Item)
        cat_time_stats = (
            hist_clean
            .join(item_cats.lazy(), on="item_id", how="inner")
            .group_by(["customer_id", "category_l3"])
            .agg([
                pl.col("created_date").max().alias("last_cat3_date"), # Ngày mua cuối
                pl.len().alias("user_cat3_buy_count") # Số lần mua Cat này
            ])
            .join(df_user_rfm.lazy().select(["customer_id", "user_purchase_count"]), on="customer_id", how="left")
            .with_columns(
                (pl.col("user_cat3_buy_count").fill_null(0) / (pl.col("user_purchase_count").fill_null(0) + 1)).alias("user_cat3_purchase_share")
            )
            .drop("user_purchase_count")
            .collect()
        )

    print("   -> Computing v13 replenishment and velocity features...")
    df_user_item_repurchase = (
        hist_clean
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("user_item_purchase_count"),
            pl.col("created_date").min().alias("first_item_purchase_date"),
            pl.col("created_date").max().alias("last_item_purchase_date"),
        ])
        .with_columns([
            (pl.lit(anchor_date) - pl.col("last_item_purchase_date"))
            .dt.total_days()
            .cast(pl.Float32)
            .alias("days_since_last_item_purchase"),
            pl.when(pl.col("user_item_purchase_count") > 1)
            .then(
                (pl.col("last_item_purchase_date") - pl.col("first_item_purchase_date"))
                .dt.total_days()
                .cast(pl.Float32)
                / (pl.col("user_item_purchase_count") - 1)
            )
            .otherwise(None)
            .alias("user_item_avg_repurchase_days"),
        ])
        .select([
            "customer_id",
            "item_id",
            "days_since_last_item_purchase",
            "user_item_purchase_count",
            "user_item_avg_repurchase_days",
        ])
        .collect()
    )

    df_user_cat3_repurchase = None
    if item_cats is not None:
        df_user_cat3_repurchase = (
            hist_clean
            .join(item_cats.lazy(), on="item_id", how="left")
            .group_by(["customer_id", "category_l3"])
            .agg([
                pl.len().alias("cat3_purchase_count"),
                pl.col("created_date").min().alias("min_cat3_date"),
                pl.col("created_date").max().alias("max_cat3_date"),
            ])
            .with_columns(
                pl.when(pl.col("cat3_purchase_count") > 1)
                .then(
                    (pl.col("max_cat3_date") - pl.col("min_cat3_date"))
                    .dt.total_days()
                    .cast(pl.Float32)
                    / (pl.col("cat3_purchase_count") - 1)
                )
                .otherwise(None)
                .alias("user_cat3_avg_repurchase_days")
            )
            .select(["customer_id", "category_l3", "user_cat3_avg_repurchase_days"])
            .collect()
        )

    recent_30_start = anchor_date - datetime.timedelta(days=30) if anchor_date is not None else None
    prev_30_start = anchor_date - datetime.timedelta(days=60) if anchor_date is not None else None
    item_velocity_base = (
        hist_clean
        .with_columns([
            (
                (pl.col("created_date") > pl.lit(recent_30_start))
                & (pl.col("created_date") <= pl.lit(anchor_date))
            ).cast(pl.Int64).alias("is_item_recent_30"),
            (
                (pl.col("created_date") > pl.lit(prev_30_start))
                & (pl.col("created_date") <= pl.lit(recent_30_start))
            ).cast(pl.Int64).alias("is_item_prev_30"),
        ])
        .group_by("item_id")
        .agg([
            pl.col("is_item_recent_30").sum().alias("item_pop_30d"),
            pl.col("is_item_prev_30").sum().alias("item_pop_prev30d"),
        ])
        .with_columns(
            (
                (pl.col("item_pop_30d").cast(pl.Float32) + 1).log()
                / (pl.col("item_pop_prev30d").cast(pl.Float32) + 2).log()
            )
            .fill_null(0)
            .alias("item_velocity_30d")
        )
    )
    df_item_velocity = (
        item_velocity_base
        .with_columns(
            pl.when(pl.col("item_velocity_30d") < 0).then(0)
            .when(pl.col("item_velocity_30d") > 5).then(5)
            .otherwise(pl.col("item_velocity_30d"))
            .alias("item_velocity_30d")
        )
        .collect()
    )

    df_cat3_velocity = None
    if item_cats is not None:
        cat3_velocity_base = (
            hist_clean
            .join(item_cats.lazy(), on="item_id", how="left")
            .with_columns([
                (
                    (pl.col("created_date") > pl.lit(recent_30_start))
                    & (pl.col("created_date") <= pl.lit(anchor_date))
                ).cast(pl.Int64).alias("is_cat3_recent_30"),
                (
                    (pl.col("created_date") > pl.lit(prev_30_start))
                    & (pl.col("created_date") <= pl.lit(recent_30_start))
                ).cast(pl.Int64).alias("is_cat3_prev_30"),
            ])
            .group_by("category_l3")
            .agg([
                pl.col("is_cat3_recent_30").sum().alias("cat3_pop_30d"),
                pl.col("is_cat3_prev_30").sum().alias("cat3_pop_prev30d"),
            ])
            .with_columns(
                (
                    (pl.col("cat3_pop_30d").cast(pl.Float32) + 1).log()
                    / (pl.col("cat3_pop_prev30d").cast(pl.Float32) + 2).log()
                )
                .fill_null(0)
                .alias("cat3_velocity_30d")
            )
        )
        df_cat3_velocity = (
            cat3_velocity_base
            .with_columns(
                pl.when(pl.col("cat3_velocity_30d") < 0).then(0)
                .when(pl.col("cat3_velocity_30d") > 5).then(5)
                .otherwise(pl.col("cat3_velocity_30d"))
                .alias("cat3_velocity_30d")
            )
            .collect()
        )

    # 3. Item Co-occurrence
    print("   -> Computing Item-Item co-occurrence...")
    co_purchase = (
        hist_clean
        .join(hist_clean, on=["customer_id", "created_date"], suffix="_right")
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"])
        .agg(pl.len().alias("cooc_score"))
        .filter(pl.col("cooc_score") >= cfg.min_coo)
    )
    df_cooc = co_purchase.collect()
    df_cooc_rev = df_cooc.select([pl.col("item_id_right").alias("item_id"), pl.col("item_id").alias("item_id_right"), pl.col("cooc_score")])
    df_cooc = pl.concat([df_cooc, df_cooc_rev]).unique(subset=["item_id", "item_id_right"])

    # 4. Brand Co-occurrence
    print("   -> Computing Brand-Brand co-occurrence...")
    df_brand_cooc = None
    if item_lf is not None and "brand" in item_schema:
        item_brands = item_lf.select(["item_id", "brand"]).unique()
        hist_with_brand = hist_clean.join(item_brands, on="item_id")
        user_brand_unique = hist_with_brand.select(["customer_id", "brand"]).unique()
        
        brand_cooc_lazy = (
            user_brand_unique
            .join(user_brand_unique, on="customer_id", suffix="_right")
            .filter(pl.col("brand") != pl.col("brand_right"))
            .group_by(["brand", "brand_right"])
            .agg(pl.len().alias("brand_cooc_score"))
            .filter(pl.col("brand_cooc_score") >= 5)
        )
        df_brand_cooc = brand_cooc_lazy.collect()
        df_brand_cooc_rev = df_brand_cooc.select([pl.col("brand_right").alias("brand"), pl.col("brand").alias("brand_right"), pl.col("brand_cooc_score")])
        df_brand_cooc = pl.concat([df_brand_cooc, df_brand_cooc_rev]).unique(subset=["brand", "brand_right"])

    # 5. User History
    print("   -> Preparing user history...")
    df_hist_long = hist_clean.sort("created_date", descending=True).group_by("customer_id").head(30).select(["customer_id", "item_id"]).rename({"item_id": "hist_item_id"}).collect()

    # 6. User Profile
    print("   -> Preparing user profile summaries...")
    if item_lf is not None:
        cols_needed = ["item_id", "brand", "category_l3", "category"]
        available_cols = item_schema
        selected_cols = [c for c in cols_needed if c in available_cols]
        item_small = item_lf.select(selected_cols)
        if "brand" not in available_cols: item_small = item_small.with_columns(pl.lit("unknown").alias("brand"))
        if "category_l3" not in available_cols: item_small = item_small.with_columns(pl.lit("unknown").alias("category_l3"))
        if "category" not in available_cols: item_small = item_small.with_columns(pl.lit("unknown").alias("category"))
        data_with_info = hist_clean.join(item_small, on="item_id", how="left")
    else:
        data_with_info = hist_clean.with_columns([pl.lit("u").alias("brand"), pl.lit("u").alias("category_l3"), pl.lit("u").alias("category")])

    df_user_profile = data_with_info.group_by("customer_id").agg([
        pl.col("brand").drop_nulls().unique().alias("hist_brands"),
        pl.col("category_l3").drop_nulls().unique().alias("hist_cats_l3"),
        pl.col("category").drop_nulls().unique().alias("hist_cats_l4")
    ]).collect()
    
    if item_lf is not None:
        df_hist_brands_long = data_with_info.select(["customer_id", "brand"]).unique().rename({"brand": "hist_brand_right"}).collect()
    else:
        df_hist_brands_long = None

    # 7. Brand Loyalty
    print("   -> Preparing Brand Loyalty...")
    if item_lf is not None and "brand" in item_schema:
        brand_info = item_lf.select(["item_id", "brand"])
        df_brand_loyalty = (
            hist_clean
            .join(brand_info, on="item_id", how="left")
            .group_by(["customer_id", "brand"])
            .agg([
                pl.len().alias("user_brand_count"),
                pl.col("created_date").max().alias("last_brand_purchase_date")
            ])
            .join(df_user_rfm.lazy().select(["customer_id", "user_purchase_count"]), on="customer_id", how="left")
            .with_columns(
                (pl.col("user_brand_count").fill_null(0) / (pl.col("user_purchase_count").fill_null(0) + 1)).alias("user_brand_purchase_share")
            )
            .drop("user_purchase_count")
            .collect()
        )
    else:
        df_brand_loyalty = None

    # 7.25 Repeat / stickiness purchase features
    print("   -> Computing repeat/stickiness purchase features...")
    user_item_counts = hist_clean.group_by(["customer_id", "item_id"]).agg(pl.len().alias("item_purchase_count_by_user"))
    df_user_stickiness = (
        user_item_counts
        .group_by("customer_id")
        .agg(
            pl.when(pl.col("item_purchase_count_by_user") > 1)
            .then(pl.col("item_purchase_count_by_user") - 1)
            .otherwise(0)
            .sum()
            .alias("repeated_item_excess")
        )
        .join(df_user_rfm.lazy().select(["customer_id", "user_purchase_count"]), on="customer_id", how="left")
        .with_columns(
            (pl.col("repeated_item_excess").fill_null(0) / (pl.col("user_purchase_count").fill_null(0) + 1)).alias("user_stickiness_score")
        )
        .select(["customer_id", "user_stickiness_score"])
        .collect()
    )

    df_user_repeat_cat3 = None
    df_cat3_repeat_rate = None
    if item_cats is not None:
        hist_with_cat = hist_clean.join(item_cats.lazy(), on="item_id", how="left")
        user_cat3_counts = hist_with_cat.group_by(["customer_id", "category_l3"]).agg(pl.len().alias("user_cat3_purchase_count"))
        df_user_repeat_cat3 = (
            user_cat3_counts
            .with_columns(
                pl.when(pl.col("user_cat3_purchase_count") > 1)
                .then(pl.col("user_cat3_purchase_count") - 1)
                .otherwise(0)
                .alias("user_cat3_repeat_excess")
            )
            .join(df_user_rfm.lazy().select(["customer_id", "user_purchase_count"]), on="customer_id", how="left")
            .with_columns(
                (pl.col("user_cat3_repeat_excess").fill_null(0) / (pl.col("user_purchase_count").fill_null(0) + 1)).alias("user_repeat_cat3_ratio")
            )
            .select(["customer_id", "category_l3", "user_repeat_cat3_ratio"])
            .collect()
        )
        df_cat3_repeat_rate = (
            hist_with_cat
            .group_by("category_l3")
            .agg([
                pl.len().alias("cat3_total_purchases"),
                pl.col("customer_id").n_unique().alias("cat3_unique_buyers"),
            ])
            .with_columns(
                (pl.col("cat3_total_purchases") / (pl.col("cat3_unique_buyers") + 1)).alias("cat3_repeat_rate")
            )
            .select(["category_l3", "cat3_repeat_rate"])
            .collect()
        )

    df_item_repeat_rate = (
        hist_clean
        .group_by("item_id")
        .agg([
            pl.len().alias("item_total_purchases"),
            pl.col("customer_id").n_unique().alias("item_unique_buyers"),
        ])
        .with_columns(
            (pl.col("item_total_purchases") / (pl.col("item_unique_buyers") + 1)).alias("item_repeat_rate")
        )
        .select(["item_id", "item_repeat_rate"])
        .collect()
    )

    # 7.5 Event Explicit Features (View / ATC)
    print("   -> Preparing explicit view/ATC features...")
    df_user_event_stats = None
    df_user_item_event_stats = None
    df_item_conversion = None
    df_cat3_conversion = None
    if event_lf is not None:
        event_hist = (
            event_lf
            .filter(pl.sql_expr(q_hist))
            .select(["customer_id", "item_id", "created_date", "event_type"])
            .with_columns([
                pl.col("event_type").cast(pl.Utf8).str.to_lowercase().str.replace_all("-", "_").alias("event_type")
            ])
            .filter(pl.col("event_type").is_in(["view_item", "add_to_cart"]))
        )

        event_hist_flagged = event_hist.with_columns([
            (pl.col("event_type") == "view_item").cast(pl.Int64).alias("is_view"),
            (pl.col("event_type") == "add_to_cart").cast(pl.Int64).alias("is_atc")
        ])

        df_user_event_stats = (
            event_hist_flagged
            .group_by("customer_id")
            .agg([
                pl.col("is_view").sum().alias("user_view_count"),
                pl.col("is_atc").sum().alias("user_atc_count"),
                pl.when(pl.col("is_view") == 1).then(pl.col("created_date")).otherwise(None).max().alias("last_view_date"),
                pl.when(pl.col("is_atc") == 1).then(pl.col("created_date")).otherwise(None).max().alias("last_atc_date"),
                pl.col("item_id").filter(pl.col("is_view") == 1).n_unique().alias("user_view_unique_items"),
                pl.col("item_id").filter(pl.col("is_atc") == 1).n_unique().alias("user_atc_unique_items")
            ])
            .collect()
        )

        df_user_item_event_stats = (
            event_hist_flagged
            .group_by(["customer_id", "item_id"])
            .agg([
                pl.col("is_view").sum().alias("user_item_view_count"),
                pl.col("is_atc").sum().alias("user_item_atc_count"),
                pl.when(pl.col("is_view") == 1).then(pl.col("created_date")).otherwise(None).max().alias("last_user_item_view_date"),
                pl.when(pl.col("is_atc") == 1).then(pl.col("created_date")).otherwise(None).max().alias("last_user_item_atc_date")
            ])
            .collect()
        )

        item_purchase_counts = hist_clean.group_by("item_id").agg(pl.len().alias("item_purchase_count"))
        item_event_counts = (
            event_hist_flagged
            .group_by("item_id")
            .agg([
                pl.col("is_view").sum().alias("item_view_count"),
                pl.col("is_atc").sum().alias("item_atc_count"),
            ])
        )
        df_item_conversion = (
            item_purchase_counts
            .join(item_event_counts, on="item_id", how="left")
            .with_columns([
                (
                    pl.col("item_purchase_count")
                    / (
                        pl.col("item_view_count").fill_null(0)
                        + pl.col("item_purchase_count")
                        + 1
                    )
                ).alias("item_view_to_purchase_rate"),
                (
                    pl.col("item_purchase_count")
                    / (
                        pl.col("item_atc_count").fill_null(0)
                        + pl.col("item_purchase_count")
                        + 1
                    )
                ).alias("item_atc_to_purchase_rate"),
            ])
            .select(["item_id", "item_view_to_purchase_rate", "item_atc_to_purchase_rate"])
            .collect()
        )

        if item_cats is not None:
            cat3_purchase_counts = (
                hist_clean
                .join(item_cats.lazy(), on="item_id", how="left")
                .group_by("category_l3")
                .agg(pl.len().alias("cat3_purchase_count"))
            )
            cat3_event_counts = (
                event_hist_flagged
                .join(item_cats.lazy(), on="item_id", how="left")
                .group_by("category_l3")
                .agg([
                    pl.col("is_view").sum().alias("cat3_view_count"),
                    pl.col("is_atc").sum().alias("cat3_atc_count"),
                ])
            )
            df_cat3_conversion = (
                cat3_purchase_counts
                .join(cat3_event_counts, on="category_l3", how="left")
                .with_columns([
                    (
                        pl.col("cat3_purchase_count")
                        / (
                            pl.col("cat3_view_count").fill_null(0)
                            + pl.col("cat3_purchase_count")
                            + 1
                        )
                    ).alias("cat3_view_to_purchase_rate"),
                    (
                        pl.col("cat3_purchase_count")
                        / (
                            pl.col("cat3_atc_count").fill_null(0)
                            + pl.col("cat3_purchase_count")
                            + 1
                        )
                    ).alias("cat3_atc_to_purchase_rate"),
                ])
                .select(["category_l3", "cat3_view_to_purchase_rate", "cat3_atc_to_purchase_rate"])
                .collect()
            )

    print("   [Replenishment/Velocity/Funnel] Added 16 v13 features")
    print(
        "      lookup rows: "
        f"user_item_repurchase={df_user_item_repurchase.height if df_user_item_repurchase is not None else 0}, "
        f"user_cat3_repurchase={df_user_cat3_repurchase.height if df_user_cat3_repurchase is not None else 0}, "
        f"item_velocity={df_item_velocity.height if df_item_velocity is not None else 0}, "
        f"cat3_velocity={df_cat3_velocity.height if df_cat3_velocity is not None else 0}, "
        f"item_conversion={df_item_conversion.height if df_item_conversion is not None else 0}, "
        f"cat3_conversion={df_cat3_conversion.height if df_cat3_conversion is not None else 0}"
    )

    # 8. Item Info Metadata
    df_item = None
    if item_lf is not None:
        df_item = item_lf.select(selected_cols).collect()
        if "brand" not in df_item.columns: df_item = df_item.with_columns(pl.lit("unknown").alias("brand"))
        if "category_l3" not in df_item.columns: df_item = df_item.with_columns(pl.lit("unknown").alias("category_l3"))
        if "category" not in df_item.columns: df_item = df_item.with_columns(pl.lit("unknown").alias("category"))

    return df_cooc, df_brand_cooc, df_hist_long, df_hist_brands_long, df_user_profile, \
            df_item_stats, df_brand_loyalty, anchor_date, df_item, user_spending, item_prices, cat_time_stats, df_user_event_stats, df_user_item_event_stats, \
            df_user_rfm, df_user_stickiness, df_user_repeat_cat3, df_item_repeat_rate, df_cat3_repeat_rate, \
            df_user_item_repurchase, df_user_cat3_repurchase, df_item_velocity, df_cat3_velocity, df_item_conversion, df_cat3_conversion

# --- 2. VECTORIZED GENERATION ---
def generate_features(
    candidates_df,
    transaction_lf,
    item_lf,
    queries,
    cfg,
    model=None,
    feature_cols=None,
    mode_name=None,
    event_lf=None,
    feature_history_query=None,
):
    print(f"   [Input] Candidates Size: {candidates_df.height} rows")
    
    mode_name = mode_name or ("inference" if model is not None else "train")
    temp_dir = f"temp_features_{mode_name}"
    success_flag = os.path.join(temp_dir, "_SUCCESS")
    
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    else:
        if os.path.exists(success_flag):
            print(f"   ✅ Found completion flag (_SUCCESS). Skipping generation step!")
            return pl.scan_parquet(f"{temp_dir}/part_*.parquet")
        print(f"   [Info] Directory {temp_dir} exists (but no success flag). Resuming...")
    
    if not hasattr(cfg, 'q_recent'): cfg.q_recent = queries['recent']
    q_hist = feature_history_query or queries['history']
    print(f"   [Feature History] {q_hist}")

    try:
        txn_schema = transaction_lf.collect_schema()
        cast_exprs = []
        if "customer_id" in txn_schema.names():
            cast_exprs.append(pl.col("customer_id").cast(txn_schema["customer_id"], strict=False))
        if "item_id" in txn_schema.names():
            cast_exprs.append(pl.col("item_id").cast(txn_schema["item_id"], strict=False))
        if cast_exprs:
            candidates_df = candidates_df.with_columns(cast_exprs)
    except Exception as e:
        print(f"   Warning: candidate key dtype alignment skipped: {e}")

    for col, default_value in CANDIDATE_FEATURE_DEFAULTS.items():
        if col not in candidates_df.columns:
            candidates_df = candidates_df.with_columns(pl.lit(default_value).alias(col))
    
    # Unpack ALL tables
    df_cooc, df_brand_cooc, df_hist_long, df_hist_brands_long, df_user_profile, \
    df_item_stats, df_brand_loyalty, anchor_date, df_item, user_spending, item_prices, cat_time_stats, df_user_event_stats, df_user_item_event_stats, \
    df_user_rfm, df_user_stickiness, df_user_repeat_cat3, df_item_repeat_rate, df_cat3_repeat_rate, \
    df_user_item_repurchase, df_user_cat3_repurchase, df_item_velocity, df_cat3_velocity, df_item_conversion, df_cat3_conversion = \
        prepare_lookup_tables(transaction_lf, item_lf, q_hist, cfg, event_lf=event_lf)
    print(f"   [Marketing Features] Added {len(MARKETING_FEATURE_COLS)} purchase-only RFM/affinity/stickiness features")

    batch_size = 1000000 
    total_rows = candidates_df.height
    pbar = tqdm(total=total_rows, desc=f"Vectorized {mode_name}")
    cand_lazy = candidates_df.lazy()

    pred_expr = None
    if model is not None and get_prediction_expr is not None:
        try:
            pred_expr = get_prediction_expr(model, feature_cols)
        except: pass

    for offset in range(0, total_rows, batch_size):
        file_name = f"{temp_dir}/part_{offset}.parquet"
        if os.path.exists(file_name):
            remaining = total_rows - offset
            step = batch_size if remaining > batch_size else remaining
            pbar.update(step)
            continue 

        chunk = cand_lazy.slice(offset, batch_size).collect()
        
        # 1. Metadata Join
        if df_item is not None:
            chunk_final = chunk.join(df_item, on="item_id", how="left")
        else:
            chunk_final = chunk.with_columns([pl.lit("u").alias("brand"), pl.lit("u").alias("category_l3"), pl.lit("u").alias("category")])

        # 2. Co-occurrence
        chunk_cooc = chunk.join(df_hist_long, on="customer_id", how="left")
        chunk_cooc = chunk_cooc.join(df_cooc, left_on=["item_id", "hist_item_id"], right_on=["item_id", "item_id_right"], how="left")
        feat_cooc = chunk_cooc.group_by(["customer_id", "item_id"]).agg([
            (pl.col("cooc_score").max().fill_null(0) + 1).log10().alias("cooc_max"),
            pl.col("cooc_score").mean().fill_null(0).alias("cooc_mean"),
            pl.col("cooc_score").sum().fill_null(0).alias("cooc_sum"),
            pl.col("cooc_score").drop_nulls().len().alias("cooc_len")
        ])
        chunk_final = chunk_final.join(feat_cooc, on=["customer_id", "item_id"], how="left")

        # 3. Brand Cross-sell
        if df_brand_cooc is not None and df_hist_brands_long is not None:
            chunk_brand_cross = chunk_final.select(["customer_id", "item_id", "brand"]).join(df_hist_brands_long, on="customer_id", how="left")
            chunk_brand_cross = chunk_brand_cross.join(df_brand_cooc, left_on=["brand", "hist_brand_right"], right_on=["brand", "brand_right"], how="left")
            feat_brand_cross = chunk_brand_cross.group_by(["customer_id", "item_id"]).agg([
                pl.col("brand_cooc_score").sum().fill_null(0).alias("brand_cross_score"), 
                pl.col("brand_cooc_score").max().fill_null(0).alias("brand_cross_max")    
            ])
            chunk_final = chunk_final.join(feat_brand_cross, on=["customer_id", "item_id"], how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0).alias("brand_cross_score"), pl.lit(0).alias("brand_cross_max")])

        # 4. Stats, Profile, Popularity
        chunk_final = chunk_final.join(df_user_profile, on="customer_id", how="left")
        
        if df_item_stats is not None:
            chunk_final = chunk_final.join(df_item_stats, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0).alias("global_item_count"))

        if df_brand_loyalty is not None:
            chunk_final = chunk_final.join(df_brand_loyalty, on=["customer_id", "brand"], how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0).alias("user_brand_count"),
                pl.lit(None).alias("last_brand_purchase_date"),
                pl.lit(0.0).alias("user_brand_purchase_share"),
            ])

        if df_user_rfm is not None:
            chunk_final = chunk_final.join(df_user_rfm, on="customer_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(999).alias("user_recency_days"),
                pl.lit(0.0).alias("user_frequency_log"),
                pl.lit(0.0).alias("user_monetary_log"),
                pl.lit(0.0).alias("user_rfm_score"),
                pl.lit(0).alias("user_purchase_count"),
            ])

        if df_user_stickiness is not None:
            chunk_final = chunk_final.join(df_user_stickiness, on="customer_id", how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0.0).alias("user_stickiness_score"))

        if df_user_event_stats is not None:
            chunk_final = chunk_final.join(df_user_event_stats, on="customer_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0).alias("user_view_count"),
                pl.lit(0).alias("user_atc_count"),
                pl.lit(None).alias("last_view_date"),
                pl.lit(None).alias("last_atc_date"),
                pl.lit(0).alias("user_view_unique_items"),
                pl.lit(0).alias("user_atc_unique_items")
            ])

        if df_user_item_event_stats is not None:
            chunk_final = chunk_final.join(df_user_item_event_stats, on=["customer_id", "item_id"], how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0).alias("user_item_view_count"),
                pl.lit(0).alias("user_item_atc_count"),
                pl.lit(None).alias("last_user_item_view_date"),
                pl.lit(None).alias("last_user_item_atc_date")
            ])

        if user_spending is not None and item_prices is not None:
            chunk_final = chunk_final.join(user_spending, on="customer_id", how="left")
            if "price" not in chunk_final.columns:
                chunk_final = chunk_final.join(item_prices, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([pl.lit(0).alias("user_avg_spend"), pl.lit(0).alias("price")])

        # [NEW] 5. Time Features (Category Cycles)
        if cat_time_stats is not None:
            # Join theo (User, Category L3) - An toàn về RAM
            chunk_final = chunk_final.join(cat_time_stats, on=["customer_id", "category_l3"], how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(None).alias("last_cat3_date"),
                pl.lit(0).alias("user_cat3_buy_count"),
                pl.lit(0.0).alias("user_cat3_purchase_share"),
            ])

        if df_user_repeat_cat3 is not None:
            chunk_final = chunk_final.join(df_user_repeat_cat3, on=["customer_id", "category_l3"], how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0.0).alias("user_repeat_cat3_ratio"))

        if df_item_repeat_rate is not None:
            chunk_final = chunk_final.join(df_item_repeat_rate, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0.0).alias("item_repeat_rate"))

        if df_cat3_repeat_rate is not None:
            chunk_final = chunk_final.join(df_cat3_repeat_rate, on="category_l3", how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(0.0).alias("cat3_repeat_rate"))

        if df_user_item_repurchase is not None:
            chunk_final = chunk_final.join(df_user_item_repurchase, on=["customer_id", "item_id"], how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(999).alias("days_since_last_item_purchase"),
                pl.lit(0).alias("user_item_purchase_count"),
                pl.lit(999).alias("user_item_avg_repurchase_days"),
            ])

        if df_user_cat3_repurchase is not None:
            chunk_final = chunk_final.join(df_user_cat3_repurchase, on=["customer_id", "category_l3"], how="left")
        else:
            chunk_final = chunk_final.with_columns(pl.lit(999).alias("user_cat3_avg_repurchase_days"))

        if df_item_velocity is not None:
            chunk_final = chunk_final.join(df_item_velocity, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0).alias("item_pop_30d"),
                pl.lit(0).alias("item_pop_prev30d"),
                pl.lit(0.0).alias("item_velocity_30d"),
            ])

        if df_cat3_velocity is not None:
            chunk_final = chunk_final.join(df_cat3_velocity, on="category_l3", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0).alias("cat3_pop_30d"),
                pl.lit(0).alias("cat3_pop_prev30d"),
                pl.lit(0.0).alias("cat3_velocity_30d"),
            ])

        if df_item_conversion is not None:
            chunk_final = chunk_final.join(df_item_conversion, on="item_id", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0.0).alias("item_view_to_purchase_rate"),
                pl.lit(0.0).alias("item_atc_to_purchase_rate"),
            ])

        if df_cat3_conversion is not None:
            chunk_final = chunk_final.join(df_cat3_conversion, on="category_l3", how="left")
        else:
            chunk_final = chunk_final.with_columns([
                pl.lit(0.0).alias("cat3_view_to_purchase_rate"),
                pl.lit(0.0).alias("cat3_atc_to_purchase_rate"),
            ])

        # 6. Calc Features
        chunk_final = chunk_final.with_columns([
            pl.col("cooc_len").fill_null(0),
            pl.col("user_brand_count").fill_null(0),
            pl.col("user_brand_purchase_share").fill_null(0),
            pl.col("brand_cross_score").fill_null(0),
            pl.col("brand_cross_max").fill_null(0),
            pl.col("global_item_count").fill_null(0),
            pl.col("price").fill_null(0),
            pl.col("user_avg_spend").fill_null(0),
            pl.col("user_recency_days").fill_null(999),
            pl.col("user_frequency_log").fill_null(0),
            pl.col("user_monetary_log").fill_null(0),
            pl.col("user_rfm_score").fill_null(0),
            pl.col("user_purchase_count").fill_null(0),
            pl.col("user_cat3_purchase_share").fill_null(0),
            pl.col("user_stickiness_score").fill_null(0),
            pl.col("user_repeat_cat3_ratio").fill_null(0),
            pl.col("item_repeat_rate").fill_null(0),
            pl.col("cat3_repeat_rate").fill_null(0),
            *[
                pl.col(c).fill_null(default)
                for c, default in V13_FEATURE_DEFAULTS.items()
                if c not in {"item_repurchase_due_score", "cat3_repurchase_due_score"}
            ],
            
            (pl.col("global_item_count") + 1).log10().alias("item_pop_log"),
            (pl.col("price") - pl.col("user_avg_spend")).abs().alias("price_diff_abs"),
            (pl.col("price") / (pl.col("user_avg_spend") + 1)).alias("price_ratio"),

            (anchor_date - pl.col("last_brand_purchase_date")).dt.total_days().fill_null(999).alias("days_since_last_brand_purchase"),
            
            (anchor_date - pl.col("last_cat3_date")).dt.total_days().fill_null(999).alias("days_since_last_cat3_purchase"),

            (pl.col("user_view_count").fill_null(0) + 1).log10().alias("user_view_count_log"),
            (pl.col("user_atc_count").fill_null(0) + 1).log10().alias("user_atc_count_log"),
            (pl.col("user_view_unique_items").fill_null(0) + 1).log10().alias("user_view_unique_items_log"),
            (pl.col("user_atc_unique_items").fill_null(0) + 1).log10().alias("user_atc_unique_items_log"),

            (anchor_date - pl.col("last_view_date")).dt.total_days().fill_null(999).alias("days_since_last_view"),
            (anchor_date - pl.col("last_atc_date")).dt.total_days().fill_null(999).alias("days_since_last_atc"),
            (pl.col("user_atc_count").fill_null(0) / (pl.col("user_view_count").fill_null(0) + 1)).alias("user_view_atc_ratio"),

            (pl.col("user_item_view_count").fill_null(0) > 0).cast(pl.Int8).alias("view_item_match"),
            (pl.col("user_item_atc_count").fill_null(0) > 0).cast(pl.Int8).alias("atc_match"),
        ])

        chunk_final = chunk_final.with_columns([
            ((pl.col("user_brand_count").fill_null(0) + 1).log() / (pl.col("days_since_last_brand_purchase").fill_null(999) + 3).log()).alias("brand_recency_x_freq"),
            ((pl.col("user_cat3_buy_count").fill_null(0) + 1).log() / (pl.col("days_since_last_cat3_purchase").fill_null(999) + 3).log()).alias("cat3_recency_x_freq"),
            pl.when((pl.col("user_item_purchase_count") > 0) & (pl.col("user_item_avg_repurchase_days") < 999))
            .then(pl.col("days_since_last_item_purchase") / (pl.col("user_item_avg_repurchase_days") + 1))
            .otherwise(0)
            .alias("_item_repurchase_due_score_raw"),
            pl.when((pl.col("user_cat3_buy_count").fill_null(0) > 0) & (pl.col("user_cat3_avg_repurchase_days") < 999))
            .then(pl.col("days_since_last_cat3_purchase") / (pl.col("user_cat3_avg_repurchase_days") + 1))
            .otherwise(0)
            .alias("_cat3_repurchase_due_score_raw"),
        ])

        chunk_final = chunk_final.with_columns([
            pl.when(pl.col("_item_repurchase_due_score_raw") < 0).then(0)
            .when(pl.col("_item_repurchase_due_score_raw") > 5).then(5)
            .otherwise(pl.col("_item_repurchase_due_score_raw"))
            .alias("item_repurchase_due_score"),
            pl.when(pl.col("_cat3_repurchase_due_score_raw") < 0).then(0)
            .when(pl.col("_cat3_repurchase_due_score_raw") > 5).then(5)
            .otherwise(pl.col("_cat3_repurchase_due_score_raw"))
            .alias("cat3_repurchase_due_score"),
        ])
        
        chunk_final = chunk_final.with_columns([
            pl.col("price_diff_abs").fill_null(99999),
            pl.col("price_ratio").fill_null(1.0),
            *[pl.col(c).fill_null(0) for c in MARKETING_FEATURE_COLS if c != "user_recency_days"],
            *[pl.col(c).fill_null(default) for c, default in V13_FEATURE_DEFAULTS.items()],
            pl.col("user_recency_days").fill_null(999),
        ])

        final_cols = [
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
            "num_candidate_sources"
        ]
        assert not any(f in final_cols for f in PRUNED_FEATURES)

        
        if model is not None:
            if pred_expr is not None:
                result_df = (
                    chunk_final
                    .select(["customer_id", "item_id"] + final_cols)
                    .with_columns([pl.col(c).cast(pl.Float32) for c in final_cols])
                    .with_columns(pred_expr.cast(pl.Float32))
                )
                cols_to_save = ["customer_id", "item_id", "pred_score"] + final_cols
                result_df.select(cols_to_save).write_parquet(file_name)
            else:
                X_test = chunk_final.select(final_cols).to_numpy()
                np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                is_ranker = model.__class__.__name__.lower().endswith("ranker")
                if hasattr(model, "predict_proba") and not is_ranker:
                    scores = model.predict_proba(X_test)[:, 1]
                else:
                    scores = model.predict(X_test)
                scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                
                cols_to_save = ["customer_id", "item_id"] + final_cols
                result_df = chunk_final.select(cols_to_save).with_columns(pl.Series("pred_score", scores).cast(pl.Float32))
                result_df.write_parquet(file_name)
        else:
            cols_to_save = ["customer_id", "item_id", "target"] + final_cols
            if "created_date" in chunk_final.columns: cols_to_save.append("created_date")
            chunk_final.select(cols_to_save).write_parquet(file_name)

        pbar.update(batch_size if offset + batch_size <= total_rows else total_rows - offset)
        del chunk_cooc, feat_cooc, chunk_final, chunk
    
    pbar.close()
    if not os.path.exists(success_flag):
        with open(success_flag, 'w') as f: f.write("done")
    
    # Cleanup
    del df_cooc, df_hist_long, df_user_profile, df_item, df_brand_loyalty, df_item_stats
    del df_user_rfm, df_user_stickiness, df_item_repeat_rate
    del df_user_item_repurchase, df_item_velocity
    if 'df_user_repeat_cat3' in locals(): del df_user_repeat_cat3
    if 'df_cat3_repeat_rate' in locals(): del df_cat3_repeat_rate
    if 'df_user_cat3_repurchase' in locals(): del df_user_cat3_repurchase
    if 'df_cat3_velocity' in locals(): del df_cat3_velocity
    if 'df_item_conversion' in locals(): del df_item_conversion
    if 'df_cat3_conversion' in locals(): del df_cat3_conversion
    if 'cat_time_stats' in locals(): del cat_time_stats
    gc.collect()

    return pl.scan_parquet(f"{temp_dir}/part_*.parquet")
