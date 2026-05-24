import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
import gc 
import os
import shutil
from datetime import timedelta

warnings.filterwarnings('ignore')

# --- 1. EXTRACT MATRIX (Giá»¯ nguyÃªn) ---
def extract_user_item_rating_coo_matrix(transaction_lf, user_mapping=None, item_mapping=None, 
                                        user_col="customer_id", item_col="item_id", 
                                        rating_col="quantity", time_col="created_date"):
    print(f"   -> Extracting interaction matrix based on {user_col}...")
    lf_filtered = (transaction_lf
                   .with_columns([(pl.col(rating_col)).alias("rating")])
                   .select([pl.col(user_col), pl.col(item_col), pl.col("rating"), pl.col(time_col)])
                   .group_by([user_col, item_col])
                   .agg([pl.col("rating").sum(), pl.col(time_col).max()])
    )

    df_all = lf_filtered.collect().to_pandas()
    df_all = df_all.dropna(subset=[user_col])

    if user_mapping is None:
        user_mapping = {id: idx for idx, id in enumerate(df_all[user_col].unique())}
    if item_mapping is None:
        item_mapping = {id: idx for idx, id in enumerate(df_all[item_col].unique())}
        
    df_all['user_idx'] = df_all[user_col].map(user_mapping)
    df_all['item_idx'] = df_all[item_col].map(item_mapping)
    
    row = df_all['item_idx'].values
    col = df_all['user_idx'].values
    data = df_all['rating'].values.astype(np.float32)
    
    df_all_coo = coo_matrix((data, (row, col)), shape=(len(item_mapping), len(user_mapping)))
    rev_user_mapping = {v: k for k, v in user_mapping.items()}
    rev_item_mapping = {v: k for k, v in item_mapping.items()}

    user_recent_history = (
        transaction_lf
        .select([pl.col(user_col), pl.col(item_col), pl.col(time_col)])
        .sort(time_col)
        .group_by(user_col, maintain_order=True)
        .agg(pl.col(item_col).tail(5).alias("recent_item_ids"))
        .collect()
    )

    user_full_history = (
        transaction_lf
        .select([pl.col(user_col), pl.col(item_col)])
        .unique()
        .group_by(user_col, maintain_order=True)
        .agg(pl.col(item_col).alias("all_item_ids"))
        .collect()
    )

    user_recent_history_map = {
        row[user_col]: row["recent_item_ids"]
        for row in user_recent_history.to_dicts()
    }

    user_full_history_map = {
        row[user_col]: row["all_item_ids"]
        for row in user_full_history.to_dicts()
    }

    return df_all_coo, user_mapping, item_mapping, rev_user_mapping, rev_item_mapping, user_recent_history_map, user_full_history_map

# --- 2. GET TRENDING ---
def get_trending_items(transaction_lf, q_hist, q_val=None, n_trend=100, include_val_for_trending=False):
    if n_trend == 0:
        return []

    if include_val_for_trending and q_val is not None:
        q_filter = f"({q_hist}) OR ({q_val})"
        print("   -> Trending Scope: History + Validation (final inference mode)")
    else:
        q_filter = q_hist
        print("   -> Trending Scope: History only (local validation mode)")

    trend_items = (
        transaction_lf
        .filter(pl.sql_expr(q_filter)) 
        .group_by("item_id")
        .agg(pl.col("created_date").len().alias("cnt"))
        .sort(["cnt", "item_id"], descending=[True, False])
        .limit(n_trend)
        .select("item_id")
        .collect()
        .get_column("item_id")
        .to_list()
    )
    return trend_items

# --- 3. TRAIN MODELS (Giá»¯ nguyÃªn) ---
def train_CF_models(transaction_lf, q_hist, recent_purchase_n=10):
    print("STAGE 1: Training models (Scikit-learn backend)...")
    
    filtered_train_lf = transaction_lf.filter(pl.sql_expr(q_hist))

    sample_df = filtered_train_lf.select(pl.col("*")).limit(1).collect()
    has_event_type = "event_type" in sample_df.columns
    if has_event_type:
        purchase_lf = filtered_train_lf.filter(
            pl.col("event_type")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
            == "purchase"
        )
    else:
        purchase_lf = filtered_train_lf
    
    print("Converting TRAIN set to Sparse Matrix...")
    df_train_cf, user_mapping, item_mapping, rev_user_mapping, rev_item_mapping, user_recent_history_map, user_full_history_map = extract_user_item_rating_coo_matrix(
        filtered_train_lf, 
        user_col="customer_id", 
        item_col="item_id", 
        rating_col="quantity",
        time_col="created_date"
    )

    df_train_csr = df_train_cf.tocsr() 
    models = []
    
    print("Training model CosineRecommender...")
    cos_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    cos_model.fit(df_train_csr)
    models.append(cos_model)
    
    print("Training model TFIDFRecommender...")
    tfidf_transformer = TfidfTransformer()
    df_train_tfidf = tfidf_transformer.fit_transform(df_train_csr)
    
    tfidf_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    tfidf_model.fit(df_train_tfidf)
    models.append(tfidf_model)

    matrices = [df_train_csr, df_train_tfidf]
    purchase_recent_history = (
        purchase_lf
        .select([pl.col("customer_id"), pl.col("item_id"), pl.col("created_date")])
        .sort("created_date")
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").tail(recent_purchase_n).alias("recent_purchase_item_ids"))
        .collect()
    )

    purchase_full_history = (
        purchase_lf
        .select([pl.col("customer_id"), pl.col("item_id")])
        .unique()
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").alias("purchase_item_ids"))
        .collect()
    )

    purchase_recent_history_map = {
        row["customer_id"]: row["recent_purchase_item_ids"]
        for row in purchase_recent_history.to_dicts()
    }

    purchase_full_history_map = {
        row["customer_id"]: row["purchase_item_ids"]
        for row in purchase_full_history.to_dicts()
    }

    return models, matrices, user_mapping, item_mapping, rev_user_mapping, rev_item_mapping, purchase_recent_history_map, purchase_full_history_map


def _build_user_signal_maps(event_lf, q_hist, recent_view_n=20, recent_atc_n=10):
    if event_lf is None:
        return {}, {}, {}

    print("   -> Building explicit view/ATC signal maps...")
    event_hist = (
        event_lf
        .filter(pl.sql_expr(q_hist))
        .select(["customer_id", "item_id", "created_date", "event_type"])
        .with_columns(
            pl.col("event_type")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
            .alias("event_type")
        )
        .filter(pl.col("event_type").is_in(["view_item", "add_to_cart"]))
    )

    view_map_df = (
        event_hist
        .filter(pl.col("event_type") == "view_item")
        .sort("created_date")
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").tail(recent_view_n).alias("view_item_ids"))
        .collect()
    )

    atc_map_df = (
        event_hist
        .filter(pl.col("event_type") == "add_to_cart")
        .sort("created_date")
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").tail(recent_atc_n).alias("atc_item_ids"))
        .collect()
    )

    view_map = {row["customer_id"]: row["view_item_ids"] for row in view_map_df.to_dicts()}
    atc_map = {row["customer_id"]: row["atc_item_ids"] for row in atc_map_df.to_dicts()}

    return event_hist, view_map, atc_map


def _build_repeatable_items(item_lf):
    if item_lf is None:
        return set()

    try:
        item_cols = item_lf.collect_schema().names()
    except Exception:
        item_cols = item_lf.limit(1).collect().columns

    text_cols = [
        col for col in [
            "category", "category_l1", "category_l2", "category_l3", "category_l4",
            "description", "item_name", "name"
        ]
        if col in item_cols
    ]
    if "item_id" not in item_cols or not text_cols:
        return set()

    keywords = [
        "tã", "bỉm", "sữa", "khăn", "ướt",
        "núm ty", "núm ti",
        "thực phẩm", "ăn dặm", "bột", "cháo", "dinh dưỡng",
        "diaper", "milk", "wipe", "nipple", "food", "nutrition",
    ]
    haystack = pl.concat_str(
        [pl.col(col).cast(pl.Utf8).fill_null("") for col in text_cols],
        separator=" "
    ).str.to_lowercase()
    repeat_expr = pl.any_horizontal([haystack.str.contains(keyword, literal=True) for keyword in keywords])

    repeatable_items = (
        item_lf
        .select([
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            repeat_expr.alias("is_repeatable"),
        ])
        .filter(pl.col("is_repeatable"))
        .select("item_id")
        .collect()
        .get_column("item_id")
        .to_list()
    )
    print(f"   -> Repeatable consumable items detected: {len(repeatable_items)}")
    return set(repeatable_items)


def _purchase_history_lf(transaction_lf, q_hist):
    hist = transaction_lf.filter(pl.sql_expr(q_hist))
    try:
        cols = hist.collect_schema().names()
    except Exception:
        cols = hist.limit(1).collect().columns
    if "event_type" in cols:
        hist = hist.filter(
            pl.col("event_type")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
            == "purchase"
        )
    return hist


def _build_category_candidate_maps(transaction_lf, item_lf, q_hist, max_user_categories=3, max_category_items=50):
    if item_lf is None:
        return {}, {}

    try:
        item_cols = item_lf.collect_schema().names()
    except Exception:
        item_cols = item_lf.limit(1).collect().columns

    category_col = next((c for c in ["category_l3", "category", "category_l4"] if c in item_cols), None)
    if "item_id" not in item_cols or category_col is None:
        return {}, {}

    print("   -> Building category-trending candidate maps...")
    item_cats = (
        item_lf
        .select([
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col(category_col).cast(pl.Utf8).fill_null("unknown").alias("category_key"),
        ])
        .filter(pl.col("category_key") != "unknown")
        .unique()
    )

    purchase_hist = (
        _purchase_history_lf(transaction_lf, q_hist)
        .select([
            pl.col("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("created_date"),
        ])
        .join(item_cats, on="item_id", how="inner")
    )

    user_cat_df = (
        purchase_hist
        .group_by(["customer_id", "category_key"])
        .agg([
            pl.len().alias("cnt"),
            pl.col("created_date").max().alias("last_date"),
        ])
        .sort(["customer_id", "cnt", "last_date", "category_key"], descending=[False, True, True, False])
        .group_by("customer_id", maintain_order=True)
        .head(max_user_categories)
        .collect()
    )
    user_category_map = {}
    for row in user_cat_df.to_dicts():
        user_category_map.setdefault(row["customer_id"], []).append(
            (row["category_key"], float(np.log1p(row["cnt"])))
        )

    cat_items_df = (
        purchase_hist
        .group_by(["category_key", "item_id"])
        .agg([
            pl.len().alias("cnt"),
            pl.col("created_date").max().alias("last_date"),
        ])
        .sort(["category_key", "cnt", "last_date", "item_id"], descending=[False, True, True, False])
        .group_by("category_key", maintain_order=True)
        .head(max_category_items)
        .collect()
    )
    category_items_map = {}
    cat_seen = {}
    for row in cat_items_df.to_dicts():
        cat = row["category_key"]
        rank = cat_seen.get(cat, 0) + 1
        cat_seen[cat] = rank
        item_score = float(np.log1p(row["cnt"])) / rank
        category_items_map.setdefault(cat, []).append((row["item_id"], item_score))

    return user_category_map, category_items_map


def _build_repeat_candidate_map(transaction_lf, q_hist, repeatable_items, max_repeat_items_per_user=20):
    if not repeatable_items:
        return {}

    print("   -> Building repeat-purchase candidate map...")
    repeat_df = (
        _purchase_history_lf(transaction_lf, q_hist)
        .select([
            pl.col("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("created_date"),
        ])
        .filter(pl.col("item_id").is_in(list(repeatable_items)))
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("cnt"),
            pl.col("created_date").max().alias("last_date"),
        ])
        .sort(["customer_id", "cnt", "last_date", "item_id"], descending=[False, True, True, False])
        .group_by("customer_id", maintain_order=True)
        .head(max_repeat_items_per_user)
        .collect()
    )

    repeat_map = {}
    for row in repeat_df.to_dicts():
        repeat_map.setdefault(row["customer_id"], []).append(
            (row["item_id"], float(np.log1p(row["cnt"])))
        )
    return repeat_map


def _build_repurchase_due_candidate_map(
    transaction_lf,
    q_hist,
    repeatable_items,
    max_repurchase_due_items_per_user=20,
    repurchase_due_min_days=7,
    repurchase_due_max_days=120,
):
    print("   -> Building repurchase-due candidate map...")
    purchase_hist = (
        _purchase_history_lf(transaction_lf, q_hist)
        .select([
            pl.col("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("created_date"),
        ])
        .filter(
            pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
    )
    anchor_df = purchase_hist.select(pl.col("created_date").max().alias("anchor")).collect()
    if anchor_df.height == 0 or anchor_df["anchor"][0] is None:
        return {}

    anchor = anchor_df["anchor"][0]
    repurchase_lf = (
        purchase_hist
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("user_item_purchase_count"),
            pl.col("created_date").min().alias("first_item_purchase_date"),
            pl.col("created_date").max().alias("last_item_purchase_date"),
        ])
        .with_columns([
            (pl.lit(anchor) - pl.col("last_item_purchase_date"))
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
            .alias("avg_repurchase_days"),
        ])
        .filter(
            (pl.col("days_since_last_item_purchase") >= repurchase_due_min_days)
            & (pl.col("days_since_last_item_purchase") <= repurchase_due_max_days)
        )
    )
    if repeatable_items:
        repurchase_lf = repurchase_lf.filter(pl.col("item_id").is_in(list(repeatable_items)))

    due_df = (
        repurchase_lf
        .with_columns([
            (pl.col("user_item_purchase_count").cast(pl.Float32) + 1).log().alias("_base_score"),
            pl.when(pl.col("avg_repurchase_days").is_not_null())
            .then(pl.col("days_since_last_item_purchase") / (pl.col("avg_repurchase_days") + 1))
            .otherwise(pl.col("days_since_last_item_purchase") / 30.0)
            .alias("_due_ratio"),
        ])
        .with_columns(
            pl.when(pl.col("avg_repurchase_days").is_not_null())
            .then(
                pl.col("_base_score")
                * pl.when(pl.col("_due_ratio") < 0).then(0)
                .when(pl.col("_due_ratio") > 3).then(3)
                .otherwise(pl.col("_due_ratio"))
            )
            .otherwise(
                pl.col("_base_score")
                * pl.when(pl.col("_due_ratio") < 0).then(0)
                .when(pl.col("_due_ratio") > 2).then(2)
                .otherwise(pl.col("_due_ratio"))
            )
            .alias("_due_score")
        )
        .with_columns(
            pl.when(pl.col("days_since_last_item_purchase") > 90)
            .then(pl.col("_due_score") * 0.7)
            .otherwise(pl.col("_due_score"))
            .alias("due_score")
        )
        .sort(["customer_id", "due_score", "item_id"], descending=[False, True, False])
        .group_by("customer_id", maintain_order=True)
        .head(max_repurchase_due_items_per_user)
        .select(["customer_id", "item_id", "due_score"])
        .collect()
    )

    due_map = {}
    for row in due_df.to_dicts():
        due_map.setdefault(row["customer_id"], []).append((row["item_id"], float(row["due_score"] or 0.0)))
    return due_map


def _build_recent_velocity_candidate_map(transaction_lf, q_hist, max_recent_velocity_items=200):
    print("   -> Building recent velocity candidate map...")
    purchase_hist = (
        _purchase_history_lf(transaction_lf, q_hist)
        .select([
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("created_date"),
        ])
        .filter(
            pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
    )
    anchor_df = purchase_hist.select(pl.col("created_date").max().alias("anchor")).collect()
    if anchor_df.height == 0 or anchor_df["anchor"][0] is None:
        return [], {}

    anchor = anchor_df["anchor"][0]
    recent_30_start = anchor - timedelta(days=30)
    prev_30_start = anchor - timedelta(days=60)
    velocity_df = (
        purchase_hist
        .with_columns([
            (
                (pl.col("created_date") > pl.lit(recent_30_start))
                & (pl.col("created_date") <= pl.lit(anchor))
            ).cast(pl.Int64).alias("is_recent_30"),
            (
                (pl.col("created_date") > pl.lit(prev_30_start))
                & (pl.col("created_date") <= pl.lit(recent_30_start))
            ).cast(pl.Int64).alias("is_prev_30"),
        ])
        .group_by("item_id")
        .agg([
            pl.col("is_recent_30").sum().alias("item_pop_30d"),
            pl.col("is_prev_30").sum().alias("item_pop_prev30d"),
        ])
        .filter(pl.col("item_pop_30d") > 0)
        .with_columns(
            (
                (pl.col("item_pop_30d").cast(pl.Float32) + 1).log()
                / (pl.col("item_pop_prev30d").cast(pl.Float32) + 2).log()
            )
            .fill_null(0)
            .alias("_item_velocity_30d")
        )
        .with_columns(
            pl.when(pl.col("_item_velocity_30d") > 5).then(5)
            .when(pl.col("_item_velocity_30d") < 0).then(0)
            .otherwise(pl.col("_item_velocity_30d"))
            .alias("item_velocity_30d")
        )
        .with_columns(
            ((pl.col("item_pop_30d").cast(pl.Float32) + 1).log() * pl.col("item_velocity_30d"))
            .alias("recent_velocity_score")
        )
        .sort(["recent_velocity_score", "item_pop_30d", "item_id"], descending=[True, True, False])
        .limit(max_recent_velocity_items)
        .select(["item_id", "recent_velocity_score"])
        .collect()
    )

    items = velocity_df.get_column("item_id").to_list() if velocity_df.height else []
    score_map = {
        row["item_id"]: float(row["recent_velocity_score"] or 0.0)
        for row in velocity_df.to_dicts()
    }
    return items, score_map


def _build_high_conversion_candidate_map(
    transaction_lf,
    event_lf,
    q_hist,
    max_high_conversion_items=200,
    min_conversion_purchase_count=5,
):
    if event_lf is None:
        return [], {}

    print("   -> Building high-conversion candidate map...")
    purchase_counts = (
        _purchase_history_lf(transaction_lf, q_hist)
        .select(pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"))
        .filter(
            pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
        .group_by("item_id")
        .agg(pl.len().alias("item_purchase_count"))
    )
    event_hist = (
        event_lf
        .filter(pl.sql_expr(q_hist))
        .select(["item_id", "event_type"])
        .with_columns([
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("event_type")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
            .alias("event_type"),
        ])
        .filter(
            pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
            & pl.col("event_type").is_in(["view_item", "add_to_cart"])
        )
        .with_columns([
            (pl.col("event_type") == "view_item").cast(pl.Int64).alias("is_view"),
            (pl.col("event_type") == "add_to_cart").cast(pl.Int64).alias("is_atc"),
        ])
        .group_by("item_id")
        .agg([
            pl.col("is_view").sum().alias("item_view_count"),
            pl.col("is_atc").sum().alias("item_atc_count"),
        ])
    )

    conversion_df = (
        purchase_counts
        .join(event_hist, on="item_id", how="left")
        .filter(pl.col("item_purchase_count") >= min_conversion_purchase_count)
        .with_columns([
            (
                pl.col("item_purchase_count")
                / (pl.col("item_view_count").fill_null(0) + pl.col("item_purchase_count") + 1)
            ).alias("item_view_to_purchase_rate"),
            (
                pl.col("item_purchase_count")
                / (pl.col("item_atc_count").fill_null(0) + pl.col("item_purchase_count") + 1)
            ).alias("item_atc_to_purchase_rate"),
        ])
        .with_columns(
            (
                (
                    0.6 * pl.col("item_atc_to_purchase_rate")
                    + 0.4 * pl.col("item_view_to_purchase_rate")
                )
                * (pl.col("item_purchase_count").cast(pl.Float32) + 1).log()
            ).alias("conversion_score")
        )
        .sort(["conversion_score", "item_purchase_count", "item_id"], descending=[True, True, False])
        .limit(max_high_conversion_items)
        .select(["item_id", "conversion_score"])
        .collect()
    )

    items = conversion_df.get_column("item_id").to_list() if conversion_df.height else []
    score_map = {
        row["item_id"]: float(row["conversion_score"] or 0.0)
        for row in conversion_df.to_dicts()
    }
    return items, score_map


def get_recent_trending_items(transaction_lf, q_hist, n_trend=100, recent_days=60):
    if n_trend == 0:
        return []

    hist = transaction_lf.filter(pl.sql_expr(q_hist))
    anchor_df = hist.select(pl.col("created_date").max().alias("anchor")).collect()
    if anchor_df.height == 0 or anchor_df["anchor"][0] is None:
        return []

    anchor = anchor_df["anchor"][0]
    recent_start = anchor - timedelta(days=recent_days)
    recent_items = (
        hist
        .filter(pl.col("created_date") > pl.lit(recent_start))
        .group_by("item_id")
        .agg(pl.len().alias("cnt"))
        .sort(["cnt", "item_id"], descending=[True, False])
        .limit(n_trend)
        .select("item_id")
        .collect()
        .get_column("item_id")
        .to_list()
    )
    return recent_items

# --- 4. PRECOMPUTE (Giá»¯ nguyÃªn) ---
def precompute_similarity_map(model, source_matrix, n_neighbors):
    n_items = source_matrix.shape[0]
    batch_size = 1000 
    sim_map = {}
    for i in range(0, n_items, batch_size):
        end = min(i + batch_size, n_items)
        batch_vectors = source_matrix[i:end]
        dists, indices = model.kneighbors(batch_vectors, n_neighbors=min(n_neighbors, n_items))
        for local_idx, neighbors in enumerate(indices):
            global_idx = i + local_idx
            valid_neighbors = [
                (int(n), max(0.0, 1.0 - float(dist)))
                for n, dist in zip(neighbors, dists[local_idx])
                if n != global_idx
            ]
            sim_map[global_idx] = valid_neighbors
    return sim_map

# --- 5. GET CANDIDATES (ÄÃƒ Sá»¬A: Ã‰p kiá»ƒu String + Tá»‘i Æ°u RAM) ---
def get_candidates(
    transaction_lf,
    q_hist,
    top_n=200,
    q_val=None,
    event_lf=None,
    item_lf=None,
    target_customers=None,
    include_val_for_trending=False,
    n_trend=200,
    neighbors_per_model=20,
    recent_purchase_n=10,
    recent_atc_n=10,
    recent_view_n=20,
    purchase_seed_weight=5.0,
    atc_seed_weight=3.0,
    view_seed_weight=1.0,
    allow_repeat_consumables=True,
    enable_category_trending=True,
    category_source_weight=2.5,
    max_user_categories=3,
    max_category_items=50,
    enable_repeat_candidates=True,
    repeat_source_weight=4.0,
    max_repeat_items_per_user=20,
    enable_recent_trending=True,
    recent_trending_days=60,
    recent_trending_weight=1.5,
    enable_repurchase_due_candidates=True,
    repurchase_due_weight=3.5,
    max_repurchase_due_items_per_user=20,
    repurchase_due_min_days=7,
    repurchase_due_max_days=120,
    enable_recent_velocity_candidates=True,
    recent_velocity_weight=1.25,
    max_recent_velocity_items=200,
    enable_high_conversion_candidates=True,
    high_conversion_weight=0.75,
    max_high_conversion_items=200,
    min_conversion_purchase_count=5,
):
    
    # Setup Temp Folder
    temp_dir = "temp_candidates"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    print(f">> Created temp dir for streaming: {temp_dir}")

    print(">> Getting Trending Items...")
    recent_trend_items_list = []
    if enable_recent_trending:
        recent_trend_items_list = get_recent_trending_items(
            transaction_lf,
            q_hist,
            n_trend=n_trend,
            recent_days=recent_trending_days,
        )
    trend_items_list = get_trending_items(
        transaction_lf,
        q_hist,
        q_val,
        n_trend=n_trend,
        include_val_for_trending=include_val_for_trending,
    )
    
    # Keep the ranked list for deterministic fallback order; use a set only for membership checks.
    recent_trend_items_list = [str(x).strip() for x in recent_trend_items_list if x is not None and str(x).strip() not in {"", "(not set)"}]
    trend_items_list = [str(x).strip() for x in trend_items_list if x is not None and str(x).strip() not in {"", "(not set)"}]
    recent_trend_items_set = set(recent_trend_items_list)
    trend_items_set = set(trend_items_list)
    repeatable_items = _build_repeatable_items(item_lf) if allow_repeat_consumables else set()
    user_category_map, category_items_map = ({}, {})
    if enable_category_trending:
        user_category_map, category_items_map = _build_category_candidate_maps(
            transaction_lf,
            item_lf,
            q_hist,
            max_user_categories=max_user_categories,
            max_category_items=max_category_items,
        )
    repeat_candidate_map = {}
    if enable_repeat_candidates:
        repeat_candidate_map = _build_repeat_candidate_map(
            transaction_lf,
            q_hist,
            repeatable_items,
            max_repeat_items_per_user=max_repeat_items_per_user,
        )
    repurchase_due_candidate_map = {}
    if enable_repurchase_due_candidates:
        repurchase_due_candidate_map = _build_repurchase_due_candidate_map(
            transaction_lf,
            q_hist,
            repeatable_items,
            max_repurchase_due_items_per_user=max_repurchase_due_items_per_user,
            repurchase_due_min_days=repurchase_due_min_days,
            repurchase_due_max_days=repurchase_due_max_days,
        )
    recent_velocity_items_list, recent_velocity_score_map = ([], {})
    if enable_recent_velocity_candidates:
        recent_velocity_items_list, recent_velocity_score_map = _build_recent_velocity_candidate_map(
            transaction_lf,
            q_hist,
            max_recent_velocity_items=max_recent_velocity_items,
        )
    high_conversion_items_list, high_conversion_score_map = ([], {})
    if enable_high_conversion_candidates:
        high_conversion_items_list, high_conversion_score_map = _build_high_conversion_candidate_map(
            transaction_lf,
            event_lf,
            q_hist,
            max_high_conversion_items=max_high_conversion_items,
            min_conversion_purchase_count=min_conversion_purchase_count,
        )

    def _is_allowed_repeat(item_id, purchased_items_str):
        if item_id not in purchased_items_str:
            return True
        return allow_repeat_consumables and item_id in repeatable_items

    trend_rank_score = {
        item: 1.0 / rank
        for rank, item in enumerate(trend_items_list, start=1)
    }
    recent_trend_rank_score = {
        item: recent_trending_weight / rank
        for rank, item in enumerate(recent_trend_items_list, start=1)
    }

    def _add_source_score(source_scores, item_id, score_name, score):
        if item_id and item_id != "(not set)":
            item_scores = source_scores.setdefault(item_id, {})
            item_scores[score_name] = max(item_scores.get(score_name, 0.0), float(score or 0.0))

    def _global_candidate_score(item):
        return max(
            recent_velocity_score_map.get(item, 0.0),
            recent_trend_rank_score.get(item, 0.0),
            high_conversion_score_map.get(item, 0.0),
            trend_rank_score.get(item, 0.0),
        )

    def _candidate_row(cust_id, item_id, rank, score, sources, source_scores=None):
        source_scores = source_scores or {}
        item_source_scores = source_scores.get(item_id, {})
        source_purchase_cf = int("purchase_cf" in sources)
        source_atc_cf = int("atc_cf" in sources)
        source_view_cf = int("view_cf" in sources)
        source_trending = int("trending" in sources)
        source_category_trending = int("category_trending" in sources)
        source_repeat_purchase = int("repeat_purchase" in sources)
        source_repurchase_due = int("repurchase_due" in sources)
        source_recent_velocity_30d = int("recent_velocity_30d" in sources)
        source_high_conversion = int("high_conversion" in sources)
        return {
            "customer_id": cust_id,
            "item_id": item_id,
            "candidate_score": float(score),
            "candidate_rank": int(rank),
            "source_purchase_cf": source_purchase_cf,
            "source_atc_cf": source_atc_cf,
            "source_view_cf": source_view_cf,
            "source_trending": source_trending,
            "source_category_trending": source_category_trending,
            "source_repeat_purchase": source_repeat_purchase,
            "source_repurchase_due": source_repurchase_due,
            "source_recent_velocity_30d": source_recent_velocity_30d,
            "source_high_conversion": source_high_conversion,
            "repurchase_due_source_score": float(item_source_scores.get("repurchase_due_source_score", 0.0)),
            "recent_velocity_source_score": float(item_source_scores.get("recent_velocity_source_score", 0.0)),
            "conversion_source_score": float(item_source_scores.get("conversion_source_score", 0.0)),
            "num_candidate_sources": (
                source_purchase_cf
                + source_atc_cf
                + source_view_cf
                + source_trending
                + source_category_trending
                + source_repeat_purchase
                + source_repurchase_due
                + source_recent_velocity_30d
                + source_high_conversion
            ),
        }

    # Train CF Model
    models, matrices, user_mapping, item_mapping, rev_user_mapping, rev_item_mapping, purchase_recent_history_map, purchase_full_history_map = train_CF_models(
        transaction_lf,
        q_hist,
        recent_purchase_n=recent_purchase_n,
    )
    _, user_view_map, user_atc_map = _build_user_signal_maps(
        event_lf,
        q_hist,
        recent_view_n=recent_view_n,
        recent_atc_n=recent_atc_n,
    )
    
    print(f">> Pre-computing Item Similarities...")
    similarity_maps = []
    for i, model in enumerate(models):
        print(f"   -> Pre-computing for model {i+1}...")
        sim_map = precompute_similarity_map(model, matrices[i], n_neighbors=neighbors_per_model)
        similarity_maps.append(sim_map)
    
    if target_customers is None:
        target_customers = list(user_mapping.keys())
    else:
        target_customers = list(target_customers)

    warm_users = sum(1 for cust_id in target_customers if cust_id in user_mapping)
    known_cold_start_users = len(target_customers) - warm_users
    print(f">> Target customers: {len(target_customers)}")
    print(f"   Warm users: {warm_users}")
    print(f"   Known cold-start users: {known_cold_start_users}")
    print(f">> Generating candidates for {len(target_customers)} customers...")
    
    batch_size = 50000 
    ncold_start = 0
    total = len(target_customers)
    
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_customers = target_customers[i:end_idx]
        batch_results = []
        
        for cust_id in batch_customers:
            is_known_user = cust_id in user_mapping

            recent_purchase_ids = purchase_recent_history_map.get(cust_id, [])
            recent_purchase_ids = recent_purchase_ids[-recent_purchase_n:] if recent_purchase_ids is not None else []
            recent_purchase_indices = [item_mapping[item_id] for item_id in recent_purchase_ids if item_id in item_mapping]

            recent_view_ids = user_view_map.get(cust_id, [])
            recent_view_ids = recent_view_ids[-recent_view_n:] if recent_view_ids is not None else []
            recent_view_indices = [item_mapping[item_id] for item_id in recent_view_ids if item_id in item_mapping]

            recent_atc_ids = user_atc_map.get(cust_id, [])
            recent_atc_ids = recent_atc_ids[-recent_atc_n:] if recent_atc_ids is not None else []
            recent_atc_indices = [item_mapping[item_id] for item_id in recent_atc_ids if item_id in item_mapping]

            purchased_items = purchase_full_history_map.get(cust_id, [])
            purchased_items_str = {str(item_id).strip() for item_id in purchased_items}
            
            # Cold start logic: includes target customers that are absent from history.
            if (not is_known_user) or (len(recent_purchase_indices) == 0 and len(recent_view_indices) == 0 and len(recent_atc_indices) == 0):
                ncold_start += 1
                cold_pool = []
                cold_seen = set()
                cold_sources = {}
                cold_source_scores = {}
                blended_global_items = (
                    recent_velocity_items_list
                    + recent_trend_items_list
                    + high_conversion_items_list
                    + trend_items_list
                )
                for item in blended_global_items:
                    if item in cold_seen or not _is_allowed_repeat(item, purchased_items_str):
                        continue
                    if item in recent_velocity_score_map:
                        cold_sources.setdefault(item, set()).add("recent_velocity_30d")
                        _add_source_score(
                            cold_source_scores,
                            item,
                            "recent_velocity_source_score",
                            recent_velocity_score_map.get(item, 0.0),
                        )
                    if item in recent_trend_items_set or item in trend_items_set:
                        cold_sources.setdefault(item, set()).add("trending")
                    if item in high_conversion_score_map:
                        cold_sources.setdefault(item, set()).add("high_conversion")
                        _add_source_score(
                            cold_source_scores,
                            item,
                            "conversion_source_score",
                            high_conversion_score_map.get(item, 0.0),
                        )
                    if item:
                        cold_pool.append(item)
                        cold_seen.add(item)
                    if len(cold_pool) >= top_n:
                        break
                final_pool = cold_pool[:top_n]
                batch_results.extend([
                    _candidate_row(
                        cust_id,
                        item,
                        rank,
                        _global_candidate_score(item),
                        cold_sources.get(item, set()),
                        cold_source_scores,
                    )
                    for rank, item in enumerate(final_pool, start=1)
                ])
                continue

            # TÃ¬m Candidates tá»« CF, giá»¯ score Ä‘á»ƒ Æ°u tiÃªn item Ä‘Æ°á»£c gá»£i Ã½ nhiá»u láº§n vÃ  tá»« lá»‹ch sá»­ gáº§n nháº¥t
            rec_pool_score = {}
            rec_pool_sources = {}
            rec_source_scores = {}
            signal_specs = [
                (recent_purchase_indices, purchase_seed_weight, "purchase_cf"),
                (recent_atc_indices, atc_seed_weight, "atc_cf"),
                (recent_view_indices, view_seed_weight, "view_cf"),
            ]

            for m_idx in range(len(models)):
                sim_map = similarity_maps[m_idx]
                for signal_indices, base_weight, source_name in signal_specs:
                    for pos, item_idx in enumerate(signal_indices):
                        if item_idx in sim_map:
                            neighbors = sim_map[item_idx]
                            recency_weight = base_weight * (1.0 + (pos * 0.15))
                            seed_item = str(rev_item_mapping[item_idx]).strip()
                            rec_pool_score[seed_item] = rec_pool_score.get(seed_item, 0.0) + recency_weight * 0.5
                            rec_pool_sources.setdefault(seed_item, set()).add(source_name)
                            for n_idx, similarity in neighbors:
                                if n_idx in rev_item_mapping:
                                    rec_item = str(rev_item_mapping[n_idx]).strip()
                                    rec_pool_score[rec_item] = rec_pool_score.get(rec_item, 0.0) + (recency_weight * similarity)
                                    rec_pool_sources.setdefault(rec_item, set()).add(source_name)

            if not rec_pool_score:
                rec_pool_score = {
                    item: _global_candidate_score(item)
                    for item in (recent_velocity_items_list + recent_trend_items_list + high_conversion_items_list + trend_items_list)
                }
                rec_pool_sources = {}
                for item in rec_pool_score:
                    if item in recent_velocity_score_map:
                        rec_pool_sources.setdefault(item, set()).add("recent_velocity_30d")
                        _add_source_score(rec_source_scores, item, "recent_velocity_source_score", recent_velocity_score_map.get(item, 0.0))
                    if item in recent_trend_items_set or item in trend_items_set:
                        rec_pool_sources.setdefault(item, set()).add("trending")
                    if item in high_conversion_score_map:
                        rec_pool_sources.setdefault(item, set()).add("high_conversion")
                        _add_source_score(rec_source_scores, item, "conversion_source_score", high_conversion_score_map.get(item, 0.0))

            if enable_repeat_candidates:
                for item, repeat_strength in repeat_candidate_map.get(cust_id, []):
                    if item and item != "(not set)":
                        rec_pool_score[item] = rec_pool_score.get(item, 0.0) + (repeat_source_weight * repeat_strength)
                        rec_pool_sources.setdefault(item, set()).add("repeat_purchase")

            if enable_repurchase_due_candidates:
                for item, due_score in repurchase_due_candidate_map.get(cust_id, []):
                    if item and item != "(not set)":
                        rec_pool_score[item] = rec_pool_score.get(item, 0.0) + (repurchase_due_weight * due_score)
                        rec_pool_sources.setdefault(item, set()).add("repurchase_due")
                        _add_source_score(rec_source_scores, item, "repurchase_due_source_score", due_score)

            if enable_category_trending:
                for category_key, user_category_strength in user_category_map.get(cust_id, []):
                    for item, item_category_score in category_items_map.get(category_key, []):
                        if item and item != "(not set)":
                            rec_pool_score[item] = rec_pool_score.get(item, 0.0) + (
                                category_source_weight * user_category_strength * item_category_score
                            )
                            rec_pool_sources.setdefault(item, set()).add("category_trending")

            if enable_recent_velocity_candidates:
                for item in recent_velocity_items_list:
                    source_score = recent_velocity_score_map.get(item, 0.0)
                    if item and item != "(not set)" and source_score > 0:
                        rec_pool_score[item] = rec_pool_score.get(item, 0.0) + (recent_velocity_weight * source_score)
                        rec_pool_sources.setdefault(item, set()).add("recent_velocity_30d")
                        _add_source_score(rec_source_scores, item, "recent_velocity_source_score", source_score)

            if enable_high_conversion_candidates:
                for item in high_conversion_items_list:
                    source_score = high_conversion_score_map.get(item, 0.0)
                    if item and item != "(not set)" and source_score > 0:
                        rec_pool_score[item] = rec_pool_score.get(item, 0.0) + (high_conversion_weight * source_score)
                        rec_pool_sources.setdefault(item, set()).add("high_conversion")
                        _add_source_score(rec_source_scores, item, "conversion_source_score", source_score)

            merged_pool = [
                item for item, _ in sorted(rec_pool_score.items(), key=lambda x: (-x[1], x[0]))
                if _is_allowed_repeat(item, purchased_items_str)
            ]

            for item in merged_pool:
                if item in recent_trend_items_set or item in trend_items_set:
                    rec_pool_sources.setdefault(item, set()).add("trending")

            if len(merged_pool) < top_n:
                seen_pool = set(merged_pool)
                fallback_items = []
                for item in (
                    recent_velocity_items_list
                    + recent_trend_items_list
                    + high_conversion_items_list
                    + trend_items_list
                ):
                    if item in seen_pool or not _is_allowed_repeat(item, purchased_items_str):
                        continue
                    fallback_items.append(item)
                    seen_pool.add(item)
                for item in fallback_items:
                    rec_pool_score.setdefault(
                        item,
                        _global_candidate_score(item),
                    )
                    if item in recent_velocity_score_map:
                        rec_pool_sources.setdefault(item, set()).add("recent_velocity_30d")
                        _add_source_score(rec_source_scores, item, "recent_velocity_source_score", recent_velocity_score_map.get(item, 0.0))
                    if item in recent_trend_items_set or item in trend_items_set:
                        rec_pool_sources.setdefault(item, set()).add("trending")
                    if item in high_conversion_score_map:
                        rec_pool_sources.setdefault(item, set()).add("high_conversion")
                        _add_source_score(rec_source_scores, item, "conversion_source_score", high_conversion_score_map.get(item, 0.0))
                merged_pool.extend(fallback_items)

            final_pool = sorted(
                merged_pool,
                key=lambda item: (
                    -rec_pool_score.get(item, trend_rank_score.get(item, 0.0)),
                    -len(rec_pool_sources.get(item, set())),
                    item,
                ),
            )[:top_n]
            batch_results.extend([
                _candidate_row(
                    cust_id,
                    item,
                    rank,
                    rec_pool_score.get(item, recent_trend_rank_score.get(item, trend_rank_score.get(item, 0.0))),
                    rec_pool_sources.get(item, set()),
                    rec_source_scores,
                )
                for rank, item in enumerate(final_pool, start=1)
            ])
        
        if batch_results:
            df_chunk = pl.DataFrame(batch_results)
            
            # [FIX] Validate that all item_ids are valid before writing
            invalid_count = df_chunk.filter(pl.col("item_id") == "(not set)").height
            if invalid_count > 0:
                print(f"   âš ï¸ Warning: Found {invalid_count} invalid item_ids, filtering them out")
                df_chunk = df_chunk.filter(pl.col("item_id") != "(not set)")
            df_chunk = df_chunk.filter(pl.col("item_id").is_not_null() & (pl.col("item_id") != ""))
            
            # Cast customer_id and item_id safely
            df_chunk = df_chunk.with_columns([
                pl.col("customer_id").cast(pl.Int64, strict=False),
                # Keep item_id as string for now; will cast in reranking with safe error handling
            ])
            df_chunk.write_parquet(f"{temp_dir}/part_{i}.parquet")
            
        del batch_results
        gc.collect()

    print(f"\nCold start users (in training set): {ncold_start}")
    
    del similarity_maps, models, matrices, user_mapping, item_mapping, purchase_recent_history_map, purchase_full_history_map
    gc.collect()
    
    print(">> Lazy Loading all chunks from disk...")
    result_df = pl.scan_parquet(f"{temp_dir}/*.parquet").collect()
    if result_df.height > 0:
        diag_cols = [
            "source_purchase_cf",
            "source_atc_cf",
            "source_view_cf",
            "source_trending",
            "source_category_trending",
            "source_repeat_purchase",
            "source_repurchase_due",
            "source_recent_velocity_30d",
            "source_high_conversion",
        ]
        source_score_cols = [
            "repurchase_due_source_score",
            "recent_velocity_source_score",
            "conversion_source_score",
        ]
        stats = result_df.select([
            pl.col("candidate_score").mean().alias("score_mean"),
            pl.col("candidate_score").min().alias("score_min"),
            pl.col("candidate_score").max().alias("score_max"),
        ] + [
            (pl.col(c).mean() * 100).alias(f"{c}_pct")
            for c in diag_cols
            if c in result_df.columns
        ] + [
            pl.col(c).mean().alias(f"{c}_mean")
            for c in source_score_cols
            if c in result_df.columns
        ] + [
            pl.col(c).max().alias(f"{c}_max")
            for c in source_score_cols
            if c in result_df.columns
        ]).row(0, named=True)
        print(">> Candidate diagnostics:")
        print(f"   candidate_score mean/min/max: {stats['score_mean']:.4f}/{stats['score_min']:.4f}/{stats['score_max']:.4f}")
        for key, value in stats.items():
            if key.endswith("_pct"):
                print(f"   {key.replace('_pct', '')}: {value:.2f}%")
        for col in source_score_cols:
            mean_key = f"{col}_mean"
            max_key = f"{col}_max"
            if mean_key in stats and max_key in stats:
                print(f"   {col} mean/max: {stats[mean_key]:.4f}/{stats[max_key]:.4f}")
    return result_df
