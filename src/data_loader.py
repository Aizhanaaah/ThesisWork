from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = {
    "1st_test": [
        "Bx_1", "By_1", "Bx_2", "By_2",
        "Bx_3", "By_3", "Bx_4", "By_4"
    ],
    "2nd_test": ["Bx_1", "By_1", "Bx_2", "By_2"],
    "3rd_test": ["Bx_1", "By_1", "Bx_2", "By_2"]
}


def load_recording(dataset_name, filename):
    file_path = DATA_DIR / dataset_name / filename

    dataset = pd.read_csv(file_path, sep="\t", header=None)
    dataset.columns = CHANNEL_NAMES[dataset_name]

    return dataset


if __name__ == "__main__":
    # Load the first recording from 3rd_test
    folder = DATA_DIR / "3rd_test"
    first_file = next(f for f in folder.iterdir() if f.is_file())

    dataset = load_recording("3rd_test", first_file.name)

    # Save the table with column names
    csv_path = OUTPUT_DIR / "test3.csv"
    dataset.to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")

    # Plot and save the figure
    ax = dataset.plot(figsize=(14, 5), title="Bearing Vibration")
    ax.set(
        xlabel="Sample index",
        ylabel="Vibration / acceleration (g)"
    )

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "vibrationpercycle3.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"Plot saved to: {plot_path}")

