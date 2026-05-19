import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
import gc 
import os
import shutil

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
        "tÃ£", "bá»‰m", "sá»¯a", "khÄƒn", "Æ°á»›t", "nÃºm", "ty", "thá»±c pháº©m",
        "Äƒn dáº·m", "bá»™t", "chÃ¡o", "dinh dÆ°á»¡ng",
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

# --- 4. PRECOMPUTE (Giá»¯ nguyÃªn) ---
def precompute_similarity_map(model, source_matrix, n_neighbors):
    n_items = source_matrix.shape[0]
    batch_size = 1000 
    sim_map = {}
    for i in range(0, n_items, batch_size):
        end = min(i + batch_size, n_items)
        batch_vectors = source_matrix[i:end]
        dists, indices = model.kneighbors(batch_vectors, n_neighbors=n_neighbors)
        for local_idx, neighbors in enumerate(indices):
            global_idx = i + local_idx
            valid_neighbors = [n for n in neighbors if n != global_idx]
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
):
    
    # Setup Temp Folder
    temp_dir = "temp_candidates"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    print(f">> Created temp dir for streaming: {temp_dir}")

    print(">> Getting Trending Items...")
    trend_items_list = get_trending_items(
        transaction_lf,
        q_hist,
        q_val,
        n_trend=n_trend,
        include_val_for_trending=include_val_for_trending,
    )
    
    # Keep the ranked list for deterministic fallback order; use a set only for membership checks.
    trend_items_list = [str(x).strip() for x in trend_items_list if x is not None]
    trend_items_set = set(trend_items_list)
    repeatable_items = _build_repeatable_items(item_lf) if allow_repeat_consumables else set()

    def _is_allowed_repeat(item_id, purchased_items_str):
        if item_id not in purchased_items_str:
            return True
        return allow_repeat_consumables and item_id in repeatable_items

    trend_rank_score = {
        item: 1.0 / rank
        for rank, item in enumerate(trend_items_list, start=1)
    }

    def _candidate_row(cust_id, item_id, rank, score, sources):
        source_purchase_cf = int("purchase_cf" in sources)
        source_atc_cf = int("atc_cf" in sources)
        source_view_cf = int("view_cf" in sources)
        source_trending = int("trending" in sources)
        return {
            "customer_id": cust_id,
            "item_id": item_id,
            "candidate_score": float(score),
            "candidate_rank": int(rank),
            "source_purchase_cf": source_purchase_cf,
            "source_atc_cf": source_atc_cf,
            "source_view_cf": source_view_cf,
            "source_trending": source_trending,
            "num_candidate_sources": (
                source_purchase_cf + source_atc_cf + source_view_cf + source_trending
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
                final_pool = [item for item in trend_items_list if _is_allowed_repeat(item, purchased_items_str)][:top_n]
                batch_results.extend([
                    _candidate_row(cust_id, item, rank, trend_rank_score.get(item, 0.0), {"trending"})
                    for rank, item in enumerate(final_pool, start=1)
                ])
                continue

            # TÃ¬m Candidates tá»« CF, giá»¯ score Ä‘á»ƒ Æ°u tiÃªn item Ä‘Æ°á»£c gá»£i Ã½ nhiá»u láº§n vÃ  tá»« lá»‹ch sá»­ gáº§n nháº¥t
            rec_pool_score = {}
            rec_pool_sources = {}
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
                            for n_idx in neighbors:
                                if n_idx in rev_item_mapping:
                                    rec_item = str(rev_item_mapping[n_idx]).strip()
                                    rec_pool_score[rec_item] = rec_pool_score.get(rec_item, 0.0) + recency_weight
                                    rec_pool_sources.setdefault(rec_item, set()).add(source_name)

            if not rec_pool_score:
                rec_pool_score = {item: 1.0 for item in trend_items_list}
                rec_pool_sources = {item: {"trending"} for item in trend_items_list}

            merged_pool = [
                item for item, _ in sorted(rec_pool_score.items(), key=lambda x: (-x[1], x[0]))
                if _is_allowed_repeat(item, purchased_items_str)
            ]

            for item in merged_pool:
                if item in trend_items_set:
                    rec_pool_sources.setdefault(item, set()).add("trending")

            if len(merged_pool) < top_n:
                seen_pool = set(merged_pool)
                fallback_items = [
                    item for item in trend_items_list
                    if item in trend_items_set and item not in seen_pool and _is_allowed_repeat(item, purchased_items_str)
                ]
                for item in fallback_items:
                    rec_pool_score.setdefault(item, trend_rank_score.get(item, 0.0))
                    rec_pool_sources.setdefault(item, set()).add("trending")
                merged_pool.extend(fallback_items)

            final_pool = sorted(
                merged_pool,
                key=lambda item: (-rec_pool_score.get(item, trend_rank_score.get(item, 0.0)), item),
            )[:top_n]
            batch_results.extend([
                _candidate_row(
                    cust_id,
                    item,
                    rank,
                    rec_pool_score.get(item, trend_rank_score.get(item, 0.0)),
                    rec_pool_sources.get(item, set()),
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
    return pl.scan_parquet(f"{temp_dir}/*.parquet").collect()
