import pickle
from pathlib import Path

import polars as pl


def _as_lazy_frame(purchase_lf_or_path):
    if isinstance(purchase_lf_or_path, (str, Path)):
        path = Path(purchase_lf_or_path)
        if path.suffix.lower() == ".parquet":
            return pl.scan_parquet(path)
        if path.suffix.lower() == ".csv":
            return pl.scan_csv(path)
        raise ValueError(f"Unsupported ground truth input file: {path}")
    if isinstance(purchase_lf_or_path, pl.DataFrame):
        return purchase_lf_or_path.lazy()
    return purchase_lf_or_path


def create_groundtruth_from_query(
    purchase_lf_or_path,
    target_query,
    output_path,
    date_col="created_date",
    event_col="event_type",
):
    """Create split-specific ground truth as {customer_id: [item_id, ...]}."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lf = _as_lazy_frame(purchase_lf_or_path)
    schema_names = lf.collect_schema().names()

    if date_col not in schema_names:
        fallback_date_col = next((c for c in ["updated_date", "created_date", "event_date"] if c in schema_names), None)
        if fallback_date_col is None:
            raise ValueError(f"Date column '{date_col}' not found. Available: {schema_names}")
        expr = pl.col(fallback_date_col)
        lf = lf.with_columns(
            pl.coalesce([
                expr.cast(pl.Datetime, strict=False),
                expr.cast(pl.Utf8).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                expr.cast(pl.Utf8).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                expr.cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime),
            ]).alias(date_col)
        )
        schema_names = lf.collect_schema().names()

    filtered = lf.filter(pl.sql_expr(target_query))

    if event_col in schema_names:
        event_norm = (
            pl.col(event_col)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all("-", "_")
        )
        filtered = filtered.filter(event_norm == "purchase")

    gt_df = (
        filtered
        .select([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
            pl.col("item_id").cast(pl.Utf8).str.strip_chars().alias("item_id"),
        ])
        .filter(
            pl.col("customer_id").is_not_null()
            & pl.col("item_id").is_not_null()
            & (pl.col("item_id") != "")
            & (pl.col("item_id") != "(not set)")
        )
        .unique()
        .group_by("customer_id")
        .agg(pl.col("item_id").sort().alias("items"))
        .collect()
    )

    gt_dict = {row["customer_id"]: row["items"] for row in gt_df.to_dicts()}

    with open(output_path, "wb") as f:
        pickle.dump(gt_dict, f)

    total_items = sum(len(items) for items in gt_dict.values())
    avg_items = total_items / len(gt_dict) if gt_dict else 0.0
    print(f"Ground truth saved: {output_path}")
    print(f"   Customers: {len(gt_dict)}")
    print(f"   Positive items: {total_items}")
    print(f"   Avg items/customer: {avg_items:.2f}")
    return gt_dict


def create_groundtruth(
    output_path=Path("data") / "groundtruth.pkl",
    transaction_path=Path("data") / "transaction_full_2025.parquet",
):
    """Backward-compatible Dec 2025 local-test ground truth helper."""
    target_query = '"created_date" > date(\'2025-11-30\') AND "created_date" <= date(\'2025-12-31\')'
    return create_groundtruth_from_query(transaction_path, target_query, output_path)


if __name__ == "__main__":
    create_groundtruth()
