
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

INPUT_FILE = (
    DATA_DIR / "processed" / "features_1st_test_all.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "processed" / "feature_quality_1st_test.csv"
)

# Only these bearings reached confirmed failure in IMS Set 1.
FAILED_BEARINGS = [3, 4]


def calculate_monotonicity(values):
    """Directional consistency, from 0 to 1."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    differences = np.diff(values)

    increasing = np.sum(differences > 0)
    decreasing = np.sum(differences < 0)

    return abs(increasing - decreasing) / (len(values) - 1)


def calculate_robustness(values, window=21):
    """Measure how closely a feature follows its smoothed trend."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan

    overall_variation = np.std(values)

    if overall_variation < 1e-12:
        return np.nan

    series = pd.Series(values)

    smoothed = series.rolling(
        window=min(window, len(series)),
        center=True,
        min_periods=5
    ).median()

    deviations = (series - smoothed).dropna()

    if deviations.empty:
        return np.nan

    fluctuation = np.std(deviations)
    robustness = np.exp(-fluctuation / overall_variation)

    return float(robustness)


def calculate_pearson_correlation(values_a, values_b):
    """Pearson correlation between two feature trajectories."""

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def calculate_trendability(features, axis, feature_name):
    """Calculate trendability for the confirmed-failure bearings."""

    columns = [
        f"{axis}_{bearing}_{feature_name}"
        for bearing in FAILED_BEARINGS
    ]

    columns = [
        column for column in columns
        if column in features.columns
    ]

    correlations = []

    for column_a, column_b in combinations(columns, 2):
        correlation = calculate_pearson_correlation(
            features[column_a],
            features[column_b]
        )

        if np.isfinite(correlation):
            correlations.append(correlation)

    if not correlations:
        return np.nan, np.nan

    trendability = min(abs(r) for r in correlations)
    direction_agreement = np.mean(
        [r > 0 for r in correlations]
    )

    return trendability, direction_agreement


def evaluate_features(features):
    results = []

    excluded = {"timestamp", "recording"}

    feature_columns = [
        column for column in features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    trendability_scores = {}

    for column in feature_columns:
        parts = column.split("_", 2)

        if len(parts) != 3:
            continue

        axis, bearing_number, feature_name = parts

        if axis not in {"Bx", "By"}:
            continue

        key = (axis, feature_name)

        if key not in trendability_scores:
            trendability_scores[key] = calculate_trendability(
                features,
                axis,
                feature_name
            )

    for column in feature_columns:
        values = features[column].to_numpy(dtype=float)
        parts = column.split("_", 2)

        if len(parts) == 3:
            axis, bearing_number, feature_name = parts

            trendability, direction_agreement = (
                trendability_scores.get(
                    (axis, feature_name),
                    (np.nan, np.nan)
                )
            )
        else:
            trendability = np.nan

        results.append({
            "feature": column,
            "monotonicity": calculate_monotonicity(values),
            "trendability": trendability,
            "robustness": calculate_robustness(values)
        })

    results = pd.DataFrame(results)

    # Sort by channel: Bx_1, By_1, Bx_2, By_2, etc.
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

    results["channel_order"] = results["feature"].apply(
        lambda name: channel_order.get(
            "_".join(name.split("_")[:2]),
            999
        )
    )

    results = results.sort_values(
        ["channel_order", "feature"]
    ).drop(columns="channel_order")

    return results


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {INPUT_FILE}\n"
            "Run feature_analysis.py first."
        )

    features = pd.read_csv(INPUT_FILE)

    results = evaluate_features(features)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.3f"
    )

    print("\nFeature quality results:")
    print(results.to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_FILE}")
