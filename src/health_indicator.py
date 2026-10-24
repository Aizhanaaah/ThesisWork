
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "feature_quality_1st_test.csv"

COMPARISON_FILE = (
    PROCESSED_DIR / "feature_comparison_1st_test.csv"
)

SELECTED_FILE = (
    PROCESSED_DIR / "selected_features_1st_test.csv"
)

# Documented selection rule:
# Keep the five highest-scoring features per sensor channel.
TOP_N_PER_CHANNEL = 5

SCORE_COLUMNS = [
    "monotonicity",
    "trendability",
    "robustness"
]


def compare_and_select_features(results):
    """Score all candidates and select the best features per channel."""

    results = results.copy()

    # Require all three scores to be available.
    results["composite_score"] = (
        results[SCORE_COLUMNS].mean(axis=1)
    )

    results["eligible"] = results[SCORE_COLUMNS].notna().all(axis=1)

    # Extract sensor channel, e.g. Bx_1 or By_3.
    results["channel"] = results["feature"].apply(
        lambda name: "_".join(name.split("_")[:2])
    )

    # Start with no features selected.
    results["selected"] = False

    eligible = results[results["eligible"]].copy()

    # Select the top five within each channel.
    for channel, group in eligible.groupby("channel", sort=False):
        top_indices = (
            group.sort_values(
                ["composite_score", "feature"],
                ascending=[False, True]
            )
            .head(TOP_N_PER_CHANNEL)
            .index
        )

        results.loc[top_indices, "selected"] = True

    # Put channels in the desired order.
    channel_order = {
        "Bx_1": 1,
        "By_1": 2,
        "Bx_2": 3,
        "By_2": 4,
        "Bx_3": 5,
        "By_3": 6,
        "Bx_4": 7,
        "By_4": 8
    }

    results["channel_order"] = results["channel"].map(
        channel_order
    ).fillna(999)

    results = results.sort_values(
        ["channel_order", "composite_score", "feature"],
        ascending=[True, False, True]
    )

    results = results.drop(columns="channel_order")

    selected = results[results["selected"]].copy()

    return results, selected


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature quality table not found: {INPUT_FILE}\n"
            "Run feature_evaluation.py first."
        )

    quality = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in ["feature"] + SCORE_COLUMNS
        if column not in quality.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    comparison, selected = compare_and_select_features(quality)

    # Save every candidate, its scores, composite score and selection status.
    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        float_format="%.3f"
    )

    # Save only the selected features.
    selected.to_csv(
        SELECTED_FILE,
        index=False,
        float_format="%.3f"
    )

    print("\nSelected features:")
    print(
        selected[
            ["feature"] + SCORE_COLUMNS + ["composite_score"]
        ].to_string(index=False)
    )

    print(f"\nCandidates evaluated: {len(comparison)}")
    print(f"Features selected: {len(selected)}")
    print(f"\nFull comparison saved to: {COMPARISON_FILE}")
    print(f"Selected features saved to: {SELECTED_FILE}")

