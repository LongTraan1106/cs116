import argparse
import os

import polars as pl

from config import load_config
from main import _standardize_source_lf, cache_tag, root_dir


DEFAULT_KS = [80, 150, 200, 300]


def _load_validation_positives(params_path, transaction_path):
    cfg = load_config(params_path)
    queries = cfg.create_query_string()
    purchase_lf = _standardize_source_lf(pl.scan_parquet(transaction_path), use_event_weights=False)

    return (
        purchase_lf
        .filter(pl.sql_expr(queries["val"]))
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        ])
        .unique()
        .collect()
    )


def _candidate_at_k(candidates_df, k):
    base = candidates_df.select([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ])

    if k is None:
        return base

    return (
        base
        .with_row_count("_candidate_order")
        .sort(["customer_id", "_candidate_order"])
        .group_by("customer_id", maintain_order=True)
        .head(k)
        .drop("_candidate_order")
    )


def compute_stage1_report(candidates_df, positives_df, ks):
    positives_df = positives_df.select([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).unique()

    total_positive_pairs = positives_df.height
    total_gt_users = positives_df.select("customer_id").unique().height

    print("\n=== Stage 1 Candidate Recall Diagnostics ===")
    print(f"Positive pairs: {total_positive_pairs}")
    print(f"GT users:       {total_gt_users}")
    print(f"Candidate rows: {candidates_df.height}")
    print(f"Candidate users:{candidates_df.select('customer_id').unique().height}")

    if candidates_df.height > 0:
        avg_candidates = (
            candidates_df
            .group_by("customer_id")
            .len()
            .select(pl.col("len").mean())
            .item()
        )
    else:
        avg_candidates = 0.0
    print(f"Avg candidates/user: {avg_candidates:.2f}")

    for k in [*ks, None]:
        cand_k = _candidate_at_k(candidates_df, k)
        hits = positives_df.join(cand_k, on=["customer_id", "item_id"], how="inner")
        hit_users = hits.select("customer_id").unique().height
        recall = hits.height / total_positive_pairs if total_positive_pairs else 0.0
        user_hit_rate = hit_users / total_gt_users if total_gt_users else 0.0
        label = f"@{k}" if k is not None else "@ALL"

        print(f"\nK{label}")
        print(f"  Candidate Recall{label}: {recall:.6f}")
        print(f"  User Hit Rate{label}:    {user_hit_rate:.6f}")
        print(f"  Hit positive pairs:      {hits.height}")
        print(f"  Users with hit:          {hit_users}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 candidate recall diagnostics.")
    parser.add_argument(
        "--candidates",
        default=f"candidates_inference_{cache_tag}.parquet",
        help="Path to Stage 1 candidate parquet.",
    )
    parser.add_argument(
        "--transactions",
        default=os.path.join(root_dir, "transaction_full_2025.parquet"),
        help="Path to purchase transaction parquet.",
    )
    parser.add_argument("--params", default="params.json", help="Path to params.json.")
    parser.add_argument(
        "--ks",
        default=",".join(str(k) for k in DEFAULT_KS),
        help="Comma-separated K values. ALL is always reported.",
    )
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    candidates_df = pl.read_parquet(args.candidates)
    positives_df = _load_validation_positives(args.params, args.transactions)
    compute_stage1_report(candidates_df, positives_df, ks)


if __name__ == "__main__":
    main()
