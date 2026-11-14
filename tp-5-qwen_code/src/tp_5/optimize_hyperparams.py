"""
Hyperparameter optimization using Optuna.

Sweeps across learning rates, batch sizes, hidden layer sizes,
and activation functions to find the best autoencoder configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .data import load_binary_dataset, create_binary_dataset
from .autoencoder import Autoencoder


def objective(trial, X_train, X_val, X_test):
    """Optuna objective function for autoencoder."""
    # Sample hyperparameters
    hidden_dims_choice = trial.suggest_categorical("hidden_dims", [
        [32],
        [64],
        [128],
        [64, 32],
        [128, 64],
        [256, 128, 64],
    ])
    activation = trial.suggest_categorical("activation", ["tanh", "sigmoid", "relu"])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_int("epochs", 200, 1500)

    # Build autoencoder
    ae = Autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=2,
        hidden_dims=hidden_dims_choice,
        activation=activation,
        lr=lr,
        optimizer="adam",
    )

    # Train
    ae.fit(
        X_train,
        X_train,
        batch_size=batch_size,
        epochs=epochs,
        print_every=200,
        validation_X=X_val,
        validation_y=X_val,
    )

    # Evaluate
    val_loss = ae.evaluate_loss(X_val, X_val)
    test_loss = ae.evaluate_loss(X_test, X_test)

    # Compute pixel error
    recon = ae.forward(X_test)
    pixel_error = np.mean(np.abs(recon - X_test) > 0.5)  # Binary threshold

    print(f"  Val loss: {val_loss:.6f}, Test loss: {test_loss:.6f}, Pixel error: {pixel_error:.4f}")

    return val_loss


def run_hyperparam_optimization(
    n_trials: int = 30,
    output_dir: str | None = None,
) -> optuna.study.Study:
    """Run hyperparameter optimization sweep."""
    # Load data
    print("Loading binary 5x7 dataset...")
    df = load_binary_dataset()

    # Convert to numpy arrays
    X = np.array([img.ravel() for img in df["image"].values], dtype=np.float64)
    labels = df["label"].values

    # Split: train / val / test
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create study
    study = optuna.create_study(direction="minimize", study_name="autoencoder_hparam")

    print(f"\n{'='*60}")
    print(f"Hyperparameter Optimization ({n_trials} trials)")
    print(f"{'='*60}")

    study.optimize(lambda trial: objective(trial, X_train, X_val, X_test), n_trials=n_trials)

    print(f"\n{'='*60}")
    print(f"Best trial: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"{'='*60}")

    # Save results
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save all trials
        trials_df = pd.DataFrame([
            {
                "trial": t.number,
                "value": t.value,
                "state": t.state.value,
                **t.params,
            }
            for t in study.trials
        ])
        trials_df.to_csv(str(out / "trials.csv"), index=False)

        # Save best trial details
        best_ae = Autoencoder(
            input_dim=X_train.shape[1],
            latent_dim=2,
            hidden_dims=study.best_params.get("hidden_dims", [64, 32]),
            activation=study.best_params.get("activation", "tanh"),
            lr=study.best_params.get("learning_rate", 0.01),
        )
        best_ae.fit(X_train, X_train, epochs=study.best_params.get("epochs", 500))
        best_val = best_ae.evaluate_loss(X_val, X_val)
        best_test = best_ae.evaluate_loss(X_test, X_test)
        best_recon = best_ae.forward(X_test)
        best_pixel_error = np.mean(np.abs(best_recon - X_test) > 0.5)

        results = {
            "best_val_loss": best_val,
            "best_test_loss": best_test,
            "best_pixel_error": best_pixel_error,
            "best_params": study.best_params,
        }
        pd.DataFrame([results]).to_csv(str(out / "best_trial.csv"), index=False)

        print(f"\nResults saved to {out}")

    return study


if __name__ == "__main__":
    study = run_hyperparam_optimization(n_trials=30)
    print(f"\nBest configuration: {study.best_params}")
