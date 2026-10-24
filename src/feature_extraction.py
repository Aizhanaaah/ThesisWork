import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_recording

# IMS vibration data sampling frequency
SAMPLING_FREQUENCY = 20_000


def extract_features(dataset):
    """Extract time-domain and frequency-domain features."""

    features = {}

    for column in dataset.columns:
        signal = dataset[column].dropna()

        if signal.empty:
            continue

        values = signal.to_numpy(dtype=float)

        # ---------- Time-domain features ----------
        mean_abs = np.mean(np.abs(values))
        rms = np.sqrt(np.mean(values ** 2))
        peak = np.max(np.abs(values))

        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_abs_mean"] = mean_abs
        features[f"{column}_rms"] = rms
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_variance"] = np.var(values)
        features[f"{column}_skewness"] = signal.skew()
        features[f"{column}_kurtosis"] = signal.kurtosis()
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_peak_to_peak"] = (
            np.max(values) - np.min(values)
        )
        features[f"{column}_crest_factor"] = (
            peak / rms if rms != 0 else 0
        )
        features[f"{column}_shape_factor"] = (
            rms / mean_abs if mean_abs != 0 else 0
        )
        features[f"{column}_impulse_factor"] = (
            peak / mean_abs if mean_abs != 0 else 0
        )

        # ---------- Frequency-domain features ----------
        # Remove the mean so the zero-frequency component
        # does not dominate the frequency analysis.
        centered_signal = values - np.mean(values)

        # Fast Fourier Transform
        fft_values = np.fft.rfft(centered_signal)
        magnitude = np.abs(fft_values)

        # Corresponding frequency values, measured in Hz
        frequencies = np.fft.rfftfreq(
            len(centered_signal),
            d=1 / SAMPLING_FREQUENCY
        )

        # Exclude the zero-frequency component
        magnitude = magnitude[1:]
        frequencies = frequencies[1:]

        if len(magnitude) > 0 and np.sum(magnitude) > 0:
            # 1. Frequency with the greatest magnitude
            dominant_index = np.argmax(magnitude)
            dominant_frequency = frequencies[dominant_index]

            # 2. Average frequency weighted by magnitude
            spectral_centroid = (
                np.sum(frequencies * magnitude)
                / np.sum(magnitude)
            )

            # 3. Frequency spread around the spectral centroid
            spectral_bandwidth = np.sqrt(
                np.sum(
                    ((frequencies - spectral_centroid) ** 2)
                    * magnitude
                ) / np.sum(magnitude)
            )

            # 4. Normalized spectral entropy (between 0 and 1)
            power = magnitude ** 2
            probabilities = power / np.sum(power)

            entropy = -np.sum(
                probabilities * np.log2(
                    probabilities + 1e-12
                )
            )

            spectral_entropy = (
                entropy / np.log2(len(probabilities))
                if len(probabilities) > 1
                else 0
            )

        else:
            dominant_frequency = 0
            spectral_centroid = 0
            spectral_bandwidth = 0
            spectral_entropy = 0

        features[f"{column}_dominant_frequency_hz"] = (
            dominant_frequency
        )
        features[f"{column}_spectral_centroid_hz"] = (
            spectral_centroid
        )
        features[f"{column}_spectral_bandwidth_hz"] = (
            spectral_bandwidth
        )
        features[f"{column}_spectral_entropy"] = (
            spectral_entropy
        )

    return pd.DataFrame([features])


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "Data"
    folder = data_dir / "1st_test"

    first_file = next(
        file for file in folder.iterdir()
        if file.is_file()
    )

    dataset = load_recording("1st_test", first_file.name)
    features = extract_features(dataset)

    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "features_test.csv"

    features.to_csv(
        output_file,
        index=False,
        float_format="%.3f"
    )

    print(features.T.to_string(header=False))
    print(f"\nSaved features to: {output_file}")