import gc
import math
import pickle

import polars as pl
from tqdm import tqdm


def _dedupe_keep_order(items, k=None):
    if items is None:
        return []
    out = []
    seen = set()
    for item in items:
        item_str = str(item).strip()
        if not item_str or item_str in seen:
            continue
        seen.add(item_str)
        out.append(item_str)
        if k is not None and len(out) >= k:
            break
    return out


def _empty_result(k, n_gt_users=0, n_pred_users=0, n_matched_users=0):
    return {
        "precision@k": 0.0,
        "recall@k": 0.0,
        "ndcg@k": 0.0,
        "map": 0.0,
        "mrr": 0.0,
        "iou": 0.0,
        "k": k,
        "n_users": n_gt_users,
        "n_gt_users": n_gt_users,
        "n_pred_users": n_pred_users,
        "n_matched_users": n_matched_users,
        "n_missing_pred_users": max(n_gt_users - n_matched_users, 0),
    }


def _score_grouped_predictions(preds, gt, k=10):
    n_gt_users = gt.height
    n_pred_users = preds.height
    merged_df = gt.join(preds, on="customer_id", how="left")
    n_matched_users = merged_df.filter(pl.col("pred_items").is_not_null()).height

    if n_gt_users == 0:
        return _empty_result(k, n_gt_users, n_pred_users, n_matched_users)

    idcg_table = {
        n_rel: sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        for n_rel in range(1, k + 1)
    }

    total_precision_at_k = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    total_iou = 0.0
    total_mrr = 0.0
    total_map = 0.0

    rows_iter = merged_df.select(["pred_items", "gt_items"]).iter_rows()
    for pred_items, gt_items in tqdm(rows_iter, total=n_gt_users, desc="Scoring"):
        pred_items = _dedupe_keep_order(pred_items, k)
        gt_items = _dedupe_keep_order(gt_items)
        gt_set = set(gt_items)
        pred_set = set(pred_items)
        if not gt_set:
            continue

        hits = 0
        dcg = 0.0
        ap = 0.0
        mrr = 0.0

        for idx, item in enumerate(pred_items):
            if item in gt_set:
                hits += 1
                dcg += 1.0 / math.log2(idx + 2)
                ap += hits / (idx + 1)
                if mrr == 0.0:
                    mrr = 1.0 / (idx + 1)

        total_precision_at_k += hits / k
        total_recall += hits / len(gt_set)
        ideal_num = min(len(gt_set), k)
        idcg = idcg_table.get(ideal_num, 0.0)
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0
        total_map += ap / ideal_num if ideal_num > 0 else 0.0
        total_mrr += mrr
        union = pred_set | gt_set
        total_iou += len(pred_set & gt_set) / len(union) if union else 0.0

    result = {
        "precision@k": total_precision_at_k / n_gt_users,
        "recall@k": total_recall / n_gt_users,
        "ndcg@k": total_ndcg / n_gt_users,
        "map": total_map / n_gt_users,
        "mrr": total_mrr / n_gt_users,
        "iou": total_iou / n_gt_users,
        "k": k,
        "n_users": n_gt_users,
        "n_gt_users": n_gt_users,
        "n_pred_users": n_pred_users,
        "n_matched_users": n_matched_users,
        "n_missing_pred_users": max(n_gt_users - n_matched_users, 0),
    }
    return result


def evaluate_predictions_df(df_pred, pos_df, k=10):
    """Evaluate validation predictions using GT users as the population."""
    print(f"\n>>> VALIDATION RANKING REPORT @ K={k}")

    gt = (
        pos_df
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
        ])
        .unique()
        .group_by("customer_id")
        .agg(pl.col("item_id").alias("gt_items"))
    )

    preds = _prepare_prediction_groups(df_pred, k)
    result = _score_grouped_predictions(preds, gt, k)
    _print_report(result)
    return result


def _prepare_prediction_groups(df_final, k):
    if df_final.height == 0:
        return pl.DataFrame({"customer_id": [], "pred_items": []}, schema={"customer_id": pl.Utf8, "pred_items": pl.List(pl.Utf8)})

    return (
        df_final
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
            pl.col("pred_score").fill_null(float("-inf")).alias("pred_score"),
        ])
        .filter(
            pl.col("customer_id").is_not_null()
            & pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
        .sort(["customer_id", "pred_score", "item_id"], descending=[False, True, False])
        .unique(subset=["customer_id", "item_id"], keep="first", maintain_order=True)
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(k).alias("pred_items"))
    )


def _load_ground_truth(gt_path):
    with open(gt_path, "rb") as f:
        gt_dict = pickle.load(f)

    rows = []
    for uid, value in gt_dict.items():
        if isinstance(value, dict) and "list_items" in value:
            items = value["list_items"]
        else:
            items = value
        rows.append([str(uid), _dedupe_keep_order(items)])

    return pl.DataFrame(rows, schema={"customer_id": pl.Utf8, "gt_items": pl.List(pl.Utf8)}, orient="row")


def _print_report(result):
    print("-" * 50)
    print(f"Evaluation users (GT): {result['n_gt_users']}")
    print(f"Prediction users:      {result['n_pred_users']}")
    print(f"Matched users:         {result['n_matched_users']}")
    print(f"Missing pred users:    {result['n_missing_pred_users']}")
    print(f"Precision@{result['k']}: {result['precision@k']:.6f}")
    print(f"Recall@{result['k']}:    {result['recall@k']:.6f}")
    print(f"NDCG@{result['k']}:      {result['ndcg@k']:.6f}")
    print(f"MAP@{result['k']}:       {result['map']:.6f}")
    print(f"MRR@{result['k']}:       {result['mrr']:.6f}")
    print(f"IoU@{result['k']}:       {result['iou']:.6f}")
    print("-" * 50)


def evaluate(df_final, k=10, gt_path="data/groundtruth.pkl", strict_gt_population=True):
    print(f"\n>>> EVALUATION @ K={k}")
    print(f">> Loading Ground Truth: {gt_path}")

    try:
        gt = _load_ground_truth(gt_path)
    except Exception as e:
        print(f"Error loading GT: {e}")
        return _empty_result(k)

    if gt.height == 0:
        print("Ground truth is empty.")
        return _empty_result(k)

    preds = _prepare_prediction_groups(df_final, k)
    if not strict_gt_population:
        gt = gt.join(preds.select("customer_id"), on="customer_id", how="inner")

    result = _score_grouped_predictions(preds, gt, k)
    _print_report(result)
    gc.collect()
    return result
