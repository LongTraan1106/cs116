import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict


@dataclass
class AppConfig:
    date_col: str
    anchor_date: str
    len_hist: int
    len_recent: int
    len_val: int
    len_test: int
    N_trend: int = 100
    N_cand: int = 20
    session_window: int = 1
    min_coo: int = 1

    def get_anchor_date(self):
        return datetime.strptime(self.anchor_date, "%Y-%m-%d").date()

    def create_query_string(self) -> Dict[str, str]:
        anchor = self.get_anchor_date()

        threshold_test = anchor - timedelta(days=self.len_test)
        threshold_val = threshold_test - timedelta(days=self.len_val)
        threshold_rec = threshold_val - timedelta(days=self.len_recent)
        threshold_hist = threshold_rec - timedelta(days=self.len_hist)

        # The standardized pipeline keeps the physical timestamp column name as
        # created_date, even when the source timestamp came from updated_date.
        c = f'"{self.date_col}"'

        queries = {
            # Backward-compatible names.
            "test": f"{c} > date('{threshold_test}') AND {c} <= date('{anchor}')",
            "val": f"{c} > date('{threshold_val}') AND {c} <= date('{threshold_test}')",
            "recent": f"{c} > date('{threshold_rec}') AND {c} <= date('{threshold_val}')",
            "history": f"{c} > date('{threshold_hist}') AND {c} <= date('{threshold_val}')",

            # Time-aware Stage 2 names.
            "train_feature_history": f"{c} > date('{threshold_hist}') AND {c} <= date('{threshold_rec}')",
            "train_target": f"{c} > date('{threshold_rec}') AND {c} <= date('{threshold_val}')",
            "val_feature_history": f"{c} > date('{threshold_hist}') AND {c} <= date('{threshold_val}')",
            "val_target": f"{c} > date('{threshold_val}') AND {c} <= date('{threshold_test}')",
            "inference_feature_history": f"{c} > date('{threshold_hist}') AND {c} <= date('{threshold_test}')",
            "val_raw_date": threshold_val,
        }

        print("\n" + "=" * 58)
        print(f"TIME SPLIT DEBUG (Anchor: {anchor}, Column: {self.date_col})")
        print("=" * 58)
        print(f"TEST TARGET:                 (> {threshold_test}) -> (<= {anchor})")
        print(f"VAL TARGET:                  (> {threshold_val}) -> (<= {threshold_test})")
        print(f"TRAIN TARGET / RECENT:       (> {threshold_rec}) -> (<= {threshold_val})")
        print("-" * 58)
        print(f"TRAIN FEATURE HISTORY:       (> {threshold_hist}) -> (<= {threshold_rec})")
        print(f"VAL FEATURE HISTORY:         (> {threshold_hist}) -> (<= {threshold_val})")
        print(f"INFERENCE FEATURE HISTORY:   (> {threshold_hist}) -> (<= {threshold_test})")
        print("Logic Check: feature windows end before their target windows.")
        print("=" * 58 + "\n")

        return queries


def load_config(file_path: str = "params.json") -> AppConfig:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find config file: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_keys = {k: v for k, v in data.items() if k in AppConfig.__annotations__}
    return AppConfig(**valid_keys)
