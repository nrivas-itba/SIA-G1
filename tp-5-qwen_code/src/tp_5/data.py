"""
Data converter: TP3 28x28 grayscale digits → 5x7 binary characters.

Also provides a clean dataset loader for the VAE (full 28x28 format).
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# Paths (absolute to avoid working directory issues)
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
TP3_DATA_DIR = _PROJECT_DIR / "tp-3" / "data"
TP5_DATA_DIR = Path(__file__).parent.parent.parent / "data"
BINARY_5x7_PATH = TP5_DATA_DIR / "binary_5x7.csv"
BINARY_5x7_NOISY_PATH = TP5_DATA_DIR / "binary_5x7_noisy.csv"


def convert_28x28_to_5x7(image_flat: np.ndarray, target_size: tuple = (5, 7)) -> np.ndarray:
    """
    Convert a 28x28 flattened image to 5x7 by dividing into patches and averaging.

    Args:
        image_flat: Flattened 784-element array (28x28)
        target_size: Target (height, width) for downscaling

    Returns:
        5x7 array with values in [0, 1]
    """
    h, w = target_size
    image = image_flat.reshape(28, 28)

    # Divide into h x w blocks and average
    block_h = 28 // h
    block_w = 28 // w

    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            y_start, y_end = i * block_h, (i + 1) * block_h
            x_start, x_end = j * block_w, (j + 1) * block_w
            result[i, j] = image[y_start:y_end, x_start:x_end].mean()

    return result


def threshold_to_binary(patch_5x7: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Threshold a 5x7 patch to binary (0 or 1).

    Args:
        patch_5x7: 5x7 array with values in [0, 1]
        threshold: Threshold value

    Returns:
        5x7 binary array (float64)
    """
    return (patch_5x7 >= threshold).astype(np.float64)


def add_binary_noise(
    binary_5x7: np.ndarray,
    noise_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add random bit-flip noise to a binary 5x7 image.

    Args:
        binary_5x7: 5x7 binary array (0 or 1)
        noise_rate: Probability of flipping each pixel
        rng: NumPy random generator for reproducibility

    Returns:
        Noisy 5x7 array
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = binary_5x7.copy()
    mask = rng.random((5, 7)) < noise_rate
    noisy[mask] = 1.0 - noisy[mask]  # Flip bits
    return noisy


def create_binary_dataset(
    source_csv: str = "digits.csv",
    target_size: tuple = (5, 7),
    threshold: float = 0.5,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Load TP3 digits.csv and convert to 5x7 binary format.

    Args:
        source_csv: Path to TP3 digits.csv
        target_size: Target (height, width) for downscaling
        threshold: Threshold for binary conversion
        output_path: Where to save the result (None = don't save)

    Returns:
        DataFrame with 'label' and 'image' (5x7 numpy arrays)
    """
    df = pd.read_csv(str(TP3_DATA_DIR / source_csv))

    images_5x7 = []
    for _, row in df.iterrows():
        img_flat = np.array(ast.literal_eval(row["image"]), dtype=np.float64)
        patch = convert_28x28_to_5x7(img_flat, target_size)
        binary = threshold_to_binary(patch, threshold)
        images_5x7.append(binary)

    result = pd.DataFrame({
        "label": df["label"].values,
        "image": images_5x7,
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both CSV and npz for reliability
        npz_path = output_path.with_suffix(".npz")
        labels = result["label"].values
        images = np.stack([img.ravel() for img in result["image"].values])
        np.savez(str(npz_path), labels=labels, images=images)
        # Also save a readable CSV (flattened images)
        df_flat = result.copy()
        df_flat["image"] = df_flat["image"].apply(lambda img: "[" + ", ".join(f"{v:.4f}" for v in img.ravel()) + "]")
        df_flat.to_csv(str(output_path), index=False)
        print(f"Saved binary 5x7 dataset to {output_path} ({len(result)} samples)")

    return result


def create_noisy_dataset(
    binary_df: pd.DataFrame,
    noise_rates: list = [0.05, 0.1, 0.15, 0.2, 0.25],
    noise_type: str = "bitflip",
    output_path: str | None = None,
) -> dict[float, pd.DataFrame]:
    """
    Create multiple noisy versions of the binary dataset.

    Args:
        binary_df: DataFrame with clean 5x7 binary images
        noise_rates: List of noise rates to generate
        noise_type: "bitflip" or "gaussian"
        output_path: Where to save (None = don't save)

    Returns:
        Dict mapping noise_rate → DataFrame with columns: label, clean_image, noisy_image
    """
    noisy_datasets = {}

    for rate in noise_rates:
        rows = []
        for _, row in binary_df.iterrows():
            clean = row["image"]
            noisy = add_binary_noise(clean, noise_rate=rate)
            rows.append({
                "label": row["label"],
                "clean_image": clean,
                "noisy_image": noisy,
            })
        df = pd.DataFrame(rows)
        noisy_datasets[rate] = df

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Save as separate files to preserve numpy arrays
            save_noisy_subset(str(out / f"noise_{rate:.2f}.csv"), df)

        print(f"  Noisy dataset (rate={rate:.2f}): {len(df)} samples")

    return noisy_datasets


def save_noisy_subset(path: str, df: pd.DataFrame, max_samples: int = 1000) -> None:
    """Save a noisy dataset subset to CSV (for visualization)."""
    subset = df.head(min(len(df), max_samples)).copy()

    def array_to_str(arr):
        return "[" + ", ".join(f"{v:.4f}" for v in arr.ravel()) + "]"

    subset["clean_image"] = subset["clean_image"].apply(array_to_str)
    subset["noisy_image"] = subset["noisy_image"].apply(array_to_str)
    subset.to_csv(path, index=False)


def load_vae_dataset(source_csv: str = "digits.csv") -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw 28x28 digits for VAE training.

    Returns:
        (X, y) where X is (n_samples, 784) normalized to [0, 1], y is labels
    """
    df = pd.read_csv(str(TP3_DATA_DIR / source_csv))
    images = np.array([np.array(ast.literal_eval(row["image"]), dtype=np.float64) for _, row in df.iterrows()])
    labels = df["label"].values
    return images, labels


def load_binary_dataset(path: str | None = None) -> pd.DataFrame:
    """Load binary 5x7 dataset from CSV or regenerate from TP3."""
    if path is None:
        path = BINARY_5x7_PATH
    path = Path(path)
    if path.exists():
        # Try reading as CSV first
        df = pd.read_csv(str(path))
        # Check if image column is already numpy arrays (from np.savez)
        sample = df["image"].iloc[0]
        if isinstance(sample, np.ndarray):
            return df
        # Handle numpy array repr saved by pandas
        try:
            df["image"] = df["image"].apply(
                lambda s: np.array(ast.literal_eval(str(s)), dtype=np.float64).reshape(5, 7)
            )
            return df
        except Exception:
            pass
        # If CSV fails, try .npz format
        npz_path = path.with_suffix(".npz")
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            images = [img.reshape(5, 7) for img in data["images"]]
            return pd.DataFrame({"label": data["labels"], "image": images})
    return create_binary_dataset(output_path=str(path))


if __name__ == "__main__":
    print("Creating binary 5x7 dataset from TP3 digits...")
    df = create_binary_dataset(output_path=str(BINARY_5x7_PATH))
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"\nSample (label=0, flattened):\n{df['image'].iloc[0].ravel()}")
