"""
Main orchestrator for TP5: runs all experiments end-to-end.

1. Data preparation (binary 5x7 from TP3 digits)
2. Standard Autoencoder training + hyperparameter optimization
3. Residual Autoencoder training + comparison
4. Denoising Autoencoder training
5. VAE training
6. All visualizations
7. py-spy performance profiling
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add tp-3 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tp-3"))

from tp_5.data import create_binary_dataset, load_binary_dataset, add_binary_noise, load_vae_dataset
from tp_5.autoencoder import Autoencoder
from tp_5.autoencoder_resnet import ResidualAutoencoder
from tp_5.denoising_autoencoder import DenoisingAutoencoder
from tp_5.vae import VariationalAutoencoder
from tp_5.optimize_hyperparams import run_hyperparam_optimization
from tp_5.visualize import (
    plot_latent_space,
    plot_reconstruction_examples,
    plot_loss_curves,
    plot_denoising_results,
    plot_vae_latent_space,
    plot_vae_reconstruction,
    plot_generated_samples,
    plot_comparison,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def prepare_data():
    """Load or create binary 5x7 dataset."""
    binary_path = Path(__file__).parent.parent / "data" / "binary_5x7.csv"
    if not binary_path.exists():
        df = create_binary_dataset(output_path=str(binary_path))
    else:
        df = load_binary_dataset(str(binary_path))

    X = np.array([img.ravel() for img in df["image"].values], dtype=np.float64)
    labels = df["label"].values
    return X, labels


def run_standard_autoencoder(X: np.ndarray, labels: np.ndarray) -> tuple[Autoencoder, dict]:
    """Train standard autoencoder with best hyperparameters."""
    print(f"\n{'='*60}")
    print("EXERCISE 1a: Standard Autoencoder")
    print(f"{'='*60}")

    # Split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    # Train with good default config
    print("\nTraining autoencoder with optimized hyperparameters...")
    ae = Autoencoder(
        input_dim=35,
        latent_dim=2,
        hidden_dims=[128, 64],
        activation="tanh",
        lr=0.005,
        optimizer="adam",
    )

    history = ae.fit(
        X_train, X_train,
        batch_size=64,
        epochs=1000,
        print_every=100,
        validation_X=X_val,
        validation_y=X_val,
    )

    # Evaluate
    val_loss = ae.evaluate_loss(X_val, X_val)
    test_loss = ae.evaluate_loss(X_test, X_test)
    recon = ae.forward(X_test)
    pixel_error = np.mean(np.abs(recon - X_test) > 0.5)

    # Per-class accuracy
    recon_binary = (recon > 0.5).astype(np.float64)
    test_binary = (X_test > 0.5).astype(np.float64)
    per_class = {}
    for label in sorted(np.unique(labels[test_idx])):
        mask = labels[test_idx] == label
        class_acc = np.mean(np.all(recon_binary[mask] == test_binary[mask], axis=1))
        per_class[int(label)] = class_acc

    # Latent space
    latent = ae.get_latent_representation(X)

    results = {
        "val_loss": val_loss,
        "test_loss": test_loss,
        "pixel_error": pixel_error,
        "per_class_accuracy": per_class,
        "latent": latent,
        "reconstruction": recon,
        "history": history,
        "X_test": X_test,
        "labels_test": labels[test_idx],
    }

    print(f"\nResults:")
    print(f"  Validation Loss: {val_loss:.6f}")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Per-pixel error: {pixel_error:.4f}")
    print(f"  Per-class accuracy: {per_class}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "autoencoder_results.json", "w") as f:
        json.dump({
            "val_loss": val_loss,
            "test_loss": test_loss,
            "pixel_error": float(pixel_error),
            "per_class_accuracy": {str(k): float(v) for k, v in per_class.items()},
        }, f, indent=2)

    # Save latent for visualization
    np.save(RESULTS_DIR / "ae_latent.npy", latent)
    np.save(RESULTS_DIR / "ae_reconstruction.npy", recon)

    return ae, results


def run_residual_autoencoder(X: np.ndarray, labels: np.ndarray) -> tuple[ResidualAutoencoder, dict]:
    """Train residual autoencoder."""
    print(f"\n{'='*60}")
    print("EXERCISE 1a (variant): Residual Autoencoder")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    train_idx, val_idx = indices[:n_train], indices[n_train:n_train+n_val]
    X_train, X_val = X[train_idx], X[val_idx]

    print("\nTraining residual autoencoder...")
    res_ae = ResidualAutoencoder(
        input_dim=35,
        latent_dim=2,
        hidden_dims=[128, 64],
        activation="tanh",
        lr=0.005,
        residual_scale=1.0,
    )

    history = res_ae.fit(
        X_train, X_train,
        batch_size=64,
        epochs=1000,
        print_every=100,
        validation_X=X_val,
        validation_y=X_val,
    )

    val_loss = res_ae.evaluate_loss(X_val, X_val)
    test_loss = res_ae.evaluate_loss(X[val_idx], X[val_idx])
    recon = res_ae.forward(X[val_idx])
    pixel_error = np.mean(np.abs(recon - X[val_idx]) > 0.5)
    latent = res_ae.get_latent_representation(X)

    results = {
        "val_loss": val_loss,
        "test_loss": test_loss,
        "pixel_error": pixel_error,
        "latent": latent,
        "reconstruction": recon,
        "history": history,
    }

    print(f"\nResults:")
    print(f"  Validation Loss: {val_loss:.6f}")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Per-pixel error: {pixel_error:.4f}")

    # Save
    np.save(RESULTS_DIR / "resnet_latent.npy", latent)
    np.save(RESULTS_DIR / "resnet_reconstruction.npy", recon)

    return res_ae, results


def run_denoising_autoencoder(X: np.ndarray, labels: np.ndarray):
    """Train denoising autoencoder and evaluate at different noise levels."""
    print(f"\n{'='*60}")
    print("EXERCISE 1b: Denoising Autoencoder")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))

    X_train = X[indices[:n_train]]

    # Train with default noise rate
    print("\nTraining denoising autoencoder (noise_rate=0.15)...")
    dae = DenoisingAutoencoder(
        input_dim=35,
        latent_dim=2,
        hidden_dims=[128, 64],
        activation="tanh",
        lr=0.005,
    )

    dae.fit(
        X_train,
        epochs=1000,
        batch_size=64,
        noise_rate=0.15,
    )

    # Evaluate at different noise levels
    print("\nEvaluating denoising capacity at different noise levels...")
    results = dae.evaluate_at_noise_levels(X[:500])

    # Save results
    with open(RESULTS_DIR / "denoising_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    return results


def run_vae(X: np.ndarray, labels: np.ndarray):
    """Train VAE and generate samples."""
    print(f"\n{'='*60}")
    print("EXERCISE 2: Variational Autoencoder (VAE)")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))

    X_train = X[indices[:n_train]]

    # Train VAE
    print("\nTraining VAE...")
    vae = VariationalAutoencoder(
        input_dim=35,
        latent_dim=2,
        hidden_dims=[128, 64],
        activation="tanh",
        lr=0.005,
        kl_weight=1.0,
    )

    history = vae.fit(
        X_train, X_train,
        batch_size=64,
        epochs=1500,
        print_every=100,
    )

    # Evaluate
    recon = vae.forward(X[:500])[0]
    recon_loss = float(np.mean((recon - X[:500]) ** 2))
    latent = vae.get_latent_representation(X)

    # Generate new samples
    print("\nGenerating new samples from VAE...")
    generated = vae.generate(n_samples=50, seed=42)

    results = {
        "reconstruction_loss": recon_loss,
        "latent": latent,
        "generated": generated,
        "history": history,
    }

    print(f"\nResults:")
    print(f"  Reconstruction Loss: {recon_loss:.6f}")

    # Save
    np.save(RESULTS_DIR / "vae_latent.npy", latent)
    np.save(RESULTS_DIR / "vae_generated.npy", generated)

    with open(RESULTS_DIR / "vae_results.json", "w") as f:
        json.dump({
            "reconstruction_loss": recon_loss,
            "n_generated": len(generated),
        }, f, indent=2)

    return vae, results


def run_py_spy_profiling(X: np.ndarray, ae: Autoencoder, n_epochs: int = 500):
    """Profile autoencoder training with py-spy."""
    print(f"\n{'='*60}")
    print("py-spy Performance Profiling")
    print(f"{'='*60}")

    profile_script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
import numpy as np
from tp_5.data import load_binary_dataset
from tp_5.autoencoder import Autoencoder

X = np.array([img.ravel() for img in {repr(X[:1000])}], dtype=np.float64)
ae = Autoencoder(input_dim=35, latent_dim=2, hidden_dims=[128, 64], activation="tanh", lr=0.005)
ae.fit(X, X, batch_size=64, epochs={n_epochs}, print_every=100)
'''

    script_path = RESULTS_DIR / "profile_script.py"
    script_path.write_text(profile_script)

    print(f"\nProfile script written to: {script_path}")
    print("\nTo run py-spy profiling:")
    print(f"  py-spy record -o {RESULTS_DIR}/profile.svg -- python {script_path}")
    print(f"\nOr for flamegraph:")
    print(f"  py-spy record -g -- -F stack -- python {script_path}")

    # Also do a simple timing-based profiling
    print("\n--- Timing analysis ---")
    ae_test = Autoencoder(input_dim=35, latent_dim=2, hidden_dims=[128, 64], activation="tanh", lr=0.005)

    n_runs = 3
    timings = []
    for run in range(n_runs):
        start = time.time()
        ae_test.fit(X[:500], X[:500], batch_size=64, epochs=100, print_every=0)
        elapsed = time.time() - start
        timings.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s (100 epochs on 500 samples)")

    avg_time = np.mean(timings)
    std_time = np.std(timings)
    print(f"\n  Average: {avg_time:.2f}s ± {std_time:.2f}s per 100 epochs (500 samples)")
    print(f"  Estimated: {avg_time * 1000 / 500:.2f}s per epoch (500 samples)")

    # Save timing results
    with open(RESULTS_DIR / "timing_results.json", "w") as f:
        json.dump({
            "n_runs": n_runs,
            "timings": [float(t) for t in timings],
            "avg_time_per_100_epochs": float(avg_time),
            "std_time": float(std_time),
            "time_per_epoch_500_samples": float(avg_time * 1000 / 500),
        }, f, indent=2)

    return {
        "avg_time": float(avg_time),
        "std_time": float(std_time),
        "script_path": str(script_path),
    }


def generate_all_graphics(
    ae_results: dict,
    resnet_results: dict,
    denoising_results: dict,
    vae_results: dict,
    labels: np.ndarray,
    X: np.ndarray,
):
    """Generate all required graphics."""
    print(f"\n{'='*60}")
    print("Generating All Graphics")
    print(f"{'='*60}")

    # 1. Autoencoder latent space (TP5 1a3)
    plot_latent_space(
        ae_results["latent"], labels,
        title="Autoencoder Latent Space (5×7 Binary Digits)",
        output_path=str(RESULTS_DIR / "ae_latent_space.png"),
    )

    # 2. Reconstruction examples (TP5 1a4)
    plot_reconstruction_examples(
        ae_results["X_test"], ae_results["reconstruction"], ae_results["labels_test"],
        n_examples=20,
        title="Autoencoder: Original vs Reconstructed",
        output_path=str(RESULTS_DIR / "ae_reconstruction_examples.png"),
    )

    # 3. Loss curve
    plot_loss_curves(
        ae_results["history"],
        title="Autoencoder: Training Loss Curve",
        output_path=str(RESULTS_DIR / "ae_loss_curve.png"),
    )

    # 4. Denoising results
    plot_denoising_results(
        denoising_results,
        output_path=str(RESULTS_DIR / "denoising_results.png"),
    )

    # 5. VAE latent space
    plot_vae_latent_space(
        vae_results["latent"], labels,
        title="VAE Latent Space (5×7 Binary Digits)",
        output_path=str(RESULTS_DIR / "vae_latent_space.png"),
    )

    # 6. VAE reconstruction
    plot_vae_reconstruction(
        X[:50], vae_results["generated"][:50] if len(vae_results["generated"]) >= 50 else vae_results["generated"],
        n_examples=10,
        title="VAE: Reconstruction vs Generated",
        output_path=str(RESULTS_DIR / "vae_reconstruction.png"),
    )

    # 7. Generated samples
    n_gen = min(20, len(vae_results["generated"]))
    plot_generated_samples(
        vae_results["generated"][:n_gen],
        n_examples=n_gen,
        title="VAE: Generated Samples from Learned Distribution",
        output_path=str(RESULTS_DIR / "vae_generated_samples.png"),
    )

    # 8. Residual vs Standard comparison
    plot_comparison(
        standard_history=ae_results["history"],
        residual_history=resnet_results["history"],
        standard_latent=ae_results["latent"],
        residual_latent=resnet_results["latent"],
        labels=labels,
        output_dir=str(RESULTS_DIR),
    )

    # 9. VAE loss curve
    plot_loss_curves(
        vae_results["history"],
        title="VAE: Training Loss (Recon + KL)",
        output_path=str(RESULTS_DIR / "vae_loss_curve.png"),
        y_label="Loss",
    )

    print(f"\n{'='*60}")
    print(f"All graphics saved to: {RESULTS_DIR}")
    print(f"{'='*60}")

    # List all files
    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  {f.name}")


def main():
    """Run all TP5 experiments."""
    print(f"{'='*60}")
    print("TP5: Deep Learning - Autoencoders and VAEs")
    print(f"{'='*60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    print("\nPreparing data from TP3 digits dataset...")
    X, labels = prepare_data()
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features (5×7 binary)")

    # 2. Standard Autoencoder
    ae, ae_results = run_standard_autoencoder(X, labels)

    # 3. Residual Autoencoder
    resnet, resnet_results = run_residual_autoencoder(X, labels)

    # 4. Denoising Autoencoder
    denoising_results = run_denoising_autoencoder(X, labels)

    # 5. VAE
    vae, vae_results = run_vae(X, labels)

    # 6. Generate all graphics
    generate_all_graphics(ae_results, resnet_results, denoising_results, vae_results, labels, X)

    # 7. Performance profiling
    profiling = run_py_spy_profiling(X, ae, n_epochs=500)

    # 8. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = {
        "autoencoder": {
            "val_loss": float(ae_results["val_loss"]),
            "test_loss": float(ae_results["test_loss"]),
            "pixel_error": float(ae_results["pixel_error"]),
        },
        "residual_autoencoder": {
            "val_loss": float(resnet_results["val_loss"]),
            "test_loss": float(resnet_results["test_loss"]),
            "pixel_error": float(resnet_results["pixel_error"]),
        },
        "vae": {
            "reconstruction_loss": float(vae_results["reconstruction_loss"]),
        },
        "profiling": profiling,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  autoencoder_results.json")
    print(f"  vae_results.json")
    print(f"  denoising_results.json")
    print(f"  summary.json")
    print(f"  Timing: {profiling['avg_time']:.2f}s ± {profiling['std_time']:.2f}s per 100 epochs (500 samples)")

    return summary


if __name__ == "__main__":
    main()
