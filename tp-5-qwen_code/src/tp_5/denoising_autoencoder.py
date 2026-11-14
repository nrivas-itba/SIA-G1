"""
Denoising Autoencoder.

Trains the autoencoder to reconstruct clean images from corrupted inputs.
Tests different noise levels to study denoising capacity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tp-3"))

from .autoencoder import Autoencoder
from .data import add_binary_noise


def add_binary_noise_vectorized(
    binary_batch: np.ndarray,
    noise_rate: float = 0.1,
) -> np.ndarray:
    """Add random bit-flip noise to a batch of binary images (n_samples, n_features)."""
    noisy = binary_batch.copy()
    rng = np.random.default_rng()
    mask = rng.random(noisy.shape) < noise_rate
    noisy[mask] = 1.0 - noisy[mask]  # Flip bits
    return noisy


class DenoisingAutoencoder:
    """
    Autoencoder trained with corrupted inputs.

    The network learns to map noisy input → clean output,
    effectively learning robust features.
    """

    def __init__(
        self,
        input_dim: int = 35,
        latent_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
        lr: float = 0.01,
        optimizer: str = "adam",
        beta: float = 1.0,
    ):
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [64, 32],
            activation=activation,
            output_activation="sigmoid",
            lr=lr,
            optimizer=optimizer,
            beta=beta,
        )
        self.input_dim = input_dim
        self.noise_rates = [0.05, 0.1, 0.15, 0.2, 0.25]

    def fit(
        self,
        clean_X: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 64,
        lr: float | None = None,
        noise_rate: float = 0.15,
        validation_X: np.ndarray | None = None,
        validation_y: np.ndarray | None = None,
    ) -> list[float]:
        """
        Train the denoising autoencoder.

        Args:
            clean_X: Clean input data (n_samples, input_dim)
            epochs: Training epochs
            batch_size: Mini-batch size
            lr: Learning rate
            noise_rate: Default noise rate for corruption
            validation_X, validation_y: Validation data

        Returns:
            Training loss history
        """
        print(f"\n{'='*60}")
        print(f"Denoising Autoencoder Training (noise_rate={noise_rate})")
        print(f"{'='*60}")

        history = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            # Create noisy batch on-the-fly (vectorized)
            for start in range(0, len(clean_X), batch_size):
                batch_clean = clean_X[start:start + batch_size]
                batch_noisy = add_binary_noise_vectorized(batch_clean, noise_rate=noise_rate)

                y_pred = self.autoencoder.encoder.forward(batch_noisy)
                y_pred = self.autoencoder.decoder.forward(y_pred)

                loss = float(np.mean((y_pred - batch_clean) ** 2)) * len(batch_clean)
                epoch_loss += loss
                n_batches += 1

                # Manual backward (reusing autoencoder's MSE gradient)
                loss_grad = 2 * (y_pred - batch_clean) / y_pred.size

                # Backward decoder (using Autoencoder's MLP layers)
                decoder_grad = loss_grad
                for i in range(len(self.autoencoder.decoder.layers) - 1, -1, -1):
                    decoder_grad, grad_W, grad_b = self.autoencoder.decoder.layers[i].backward(decoder_grad)

                # Backward encoder (using Autoencoder's MLP layers)
                encoder_grad = decoder_grad
                for i in range(len(self.autoencoder.encoder.layers) - 1, -1, -1):
                    encoder_grad, grad_W, grad_b = self.autoencoder.encoder.layers[i].backward(encoder_grad)

            epoch_loss /= len(clean_X)
            history.append(epoch_loss)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f}")

        return history

    def evaluate_at_noise_levels(
        self,
        clean_X: np.ndarray,
        validation_X: np.ndarray | None = None,
        validation_y: np.ndarray | None = None,
    ) -> dict:
        """
        Evaluate denoising capacity at different noise levels.

        Returns:
            Dict mapping noise_rate → {loss, recon_error}
        """
        results = {}

        for rate in self.noise_rates:
            # Create noisy inputs (vectorized)
            noisy_X = add_binary_noise_vectorized(clean_X, noise_rate=rate)

            # Reconstruct (network trained to map noisy → clean)
            reconstructed = self.autoencoder.forward(noisy_X)

            # Evaluate on clean targets
            loss = float(np.mean((reconstructed - clean_X) ** 2))
            results[int(rate * 100)] = {
                "noise_rate": rate,
                "reconstruction_loss": loss,
                "mean_per_pixel_error": np.mean(np.abs(reconstructed - clean_X)),
            }

            print(f"  Noise rate {rate:.2f}: loss={loss:.6f}, per-pixel error={results[int(rate * 100)]['mean_per_pixel_error']:.6f}")

        # Also test on clean (no noise)
        clean_recon = self.autoencoder.forward(clean_X)
        clean_loss = float(np.mean((clean_recon - clean_X) ** 2))
        results[0] = {
            "noise_rate": 0.0,
            "reconstruction_loss": clean_loss,
            "mean_per_pixel_error": float(np.mean(np.abs(clean_recon - clean_X))),
        }

        return results

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent representation of clean inputs."""
        return self.autoencoder.get_latent_representation(X)
