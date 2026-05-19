"""Compatibility wrapper for candidate diagnostics.

The old test.py expected a hardcoded candidates_stage1.parquet file and an
undefined pos_df. Use stage1_diagnostics.py instead so the candidate cache and
validation split stay aligned with the current time-aware pipeline.
"""


def main():
    print("test.py is deprecated.")
    print("Use:")
    print("  python stage1_diagnostics.py --candidates candidates_val_v8_candidate_features.parquet")


if __name__ == "__main__":
    main()
