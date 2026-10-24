
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_recording
from feature_extraction import extract_features


# ---------- Paths and settings ----------

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUTPUT_DIR = DATA_DIR / "processed"
PLOT_DIR = OUTPUT_DIR / "plots"

DATASET_NAME = "1st_test"
SAMPLING_FREQUENCY = 20_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Features to track across the experiment
FEATURES_TO_PLOT = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_to_peak",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
]


def get_recording_files():
    """Return recording files ordered by timestamp."""

    folder = DATA_DIR / DATASET_NAME

    files = sorted(
        file for file in folder.iterdir()
        if file.is_file()
    )

    return files


def extract_all_recordings(files):
    """Extract features from every recording."""

    all_features = []

    for index, file in enumerate(files, start=1):
        dataset = load_recording(DATASET_NAME, file.name)

        row = extract_features(dataset)

        timestamp = pd.to_datetime(
            file.name,
            format="%Y.%m.%d.%H.%M.%S"
        )

        row.insert(0, "timestamp", timestamp)
        row.insert(1, "recording", file.name)

        all_features.append(row)

        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} recordings")

    if not all_features:
        raise ValueError("No recording files were found.")

    result = pd.concat(all_features, ignore_index=True)
    result = result.sort_values("timestamp")

    return result


def plot_feature_trends(features):
    """Plot each selected feature for each bearing."""

    bearing_channels = {
        1: ["Bx_1", "By_1"],
        2: ["Bx_2", "By_2"],
        3: ["Bx_3", "By_3"],
        4: ["Bx_4", "By_4"],
    }

    for feature_name in FEATURES_TO_PLOT:
        fig, axes = plt.subplots(
            4, 1,
            figsize=(14, 12),
            sharex=True
        )

        for bearing_number, ax in enumerate(axes, start=1):
            channels = bearing_channels[bearing_number]
            plotted = False

            for channel in channels:
                column = f"{channel}_{feature_name}"

                if column not in features.columns:
                    continue

                ax.plot(
                    features["timestamp"],
                    features[column],
                    label=channel,
                    linewidth=0.8
                )
                plotted = True

            ax.set_title(f"Bearing {bearing_number}")
            ax.set_ylabel(feature_name)
            ax.grid(True, alpha=0.3)

            if plotted:
                ax.legend()

        axes[-1].set_xlabel("Recording timestamp")

        fig.suptitle(
            f"{feature_name.replace('_', ' ').title()} "
            f"Across Bearing Lifetime"
        )

        fig.tight_layout()

        output_file = PLOT_DIR / f"{feature_name}_over_time.png"
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

        print(f"Saved plot: {output_file}")


def plot_frequency_spectra(files):
    """Compare vibration spectra at early, middle, and late stages."""

    if not files:
        return

    # Choose representative recordings from the experiment.
    selected_indices = sorted(set([
        0,
        len(files) // 2,
        len(files) - 1
    ]))

    for index in selected_indices:
        file = files[index]
        dataset = load_recording(DATASET_NAME, file.name)

        fig, axes = plt.subplots(
            4, 1,
            figsize=(14, 12),
            sharex=True
        )

        for bearing_number, ax in enumerate(axes, start=1):
            channels = [
                f"Bx_{bearing_number}",
                f"By_{bearing_number}"
            ]

            for channel in channels:
                if channel not in dataset.columns:
                    continue

                signal = dataset[channel].dropna().to_numpy(
                    dtype=float
                )

                if len(signal) == 0:
                    continue

                signal = signal - np.mean(signal)

                fft_values = np.fft.rfft(signal)
                frequencies = np.fft.rfftfreq(
                    len(signal),
                    d=1 / SAMPLING_FREQUENCY
                )

                # Amplitude spectrum for visual comparison.
                magnitude = np.abs(fft_values) / len(signal)

                ax.plot(
                    frequencies,
                    magnitude,
                    label=channel,
                    linewidth=0.8
                )

            ax.set_title(f"Bearing {bearing_number}")
            ax.set_ylabel("FFT magnitude")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")

        fig.suptitle(
            f"Vibration Frequency Spectra: {file.name}"
        )

        fig.tight_layout()

        output_file = (
            PLOT_DIR / f"spectrum_recording_{index + 1}.png"
        )

        fig.savefig(output_file, dpi=150)
        plt.close(fig)

        print(f"Saved spectrum: {output_file}")


if __name__ == "__main__":
    files = get_recording_files()

    print(f"Found {len(files)} recordings in {DATASET_NAME}")

    features = extract_all_recordings(files)

    csv_file = OUTPUT_DIR / f"features_{DATASET_NAME}_all.csv"

    features.to_csv(
        csv_file,
        index=False,
        float_format="%.3f"
    )

    print(f"Saved all features to: {csv_file}")

    plot_feature_trends(features)
    plot_frequency_spectra(files)

