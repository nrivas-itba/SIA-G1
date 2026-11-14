"""
Variational Autoencoder (VAE) built on TP3 engine.

Extends the autoencoder framework to learn a probabilistic latent
distribution p(z|x) with KL divergence regularisation.

Architecture:
    Encoder: x → h1 → (mean, log_var)  [two parallel outputs]
    Sampling: z = mean + exp(log_var/2) * epsilon,  epsilon ~ N(0,1)
    Decoder: z → h1 → reconstruction
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tp-3"))
from tp_3.engine.layer import Layer
from tp_3.engine.activation_functions.sigmoid import Sigmoid
from tp_3.engine.activation_functions.tanh import Tanh
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.optimizers.sgd import SGD
from tp_3.engine.loss_functions.mse import MeanSquaredError


class VariationalAutoencoder:
    """
    Variational Autoencoder with reparameterization trick.

    Loss = Reconstruction Loss + KL Divergence
    KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
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
        kl_weight: float = 1.0,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.activation = activation
        self.beta = beta
        self.kl_weight = kl_weight

        # Build activation map
        act_map = {
            "sigmoid": Sigmoid(beta=beta),
            "tanh": Tanh(beta=beta),
            "relu": ReLU(),
        }
        self.hidden_act = act_map[activation]

        # Encoder: input_dim → h1 → h2 → ... → latent_dim (mean) + latent_dim (log_var)
        encoder_dims_mean = [input_dim] + self.hidden_dims + [latent_dim]
        encoder_dims_logvar = [input_dim] + self.hidden_dims + [latent_dim]
        self.encoder_mean = self._build_mlp(encoder_dims_mean, self.hidden_act, None)
        self.encoder_logvar = self._build_mlp(encoder_dims_logvar, self.hidden_act, None)

        # Decoder: latent_dim → h2 → h1 → ... → input_dim
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [input_dim]
        self.decoder = self._build_mlp(decoder_dims, self.hidden_act, act_map["sigmoid"])

        # Optimizer states (lazy init in fit)
        self.encoder_mean_opt_W = None
        self.encoder_mean_opt_b = None
        self.encoder_logvar_opt_W = None
        self.encoder_logvar_opt_b = None
        self.decoder_opt_W = None
        self.decoder_opt_b = None

    def _build_mlp(
        self,
        layer_sizes: list[int],
        hidden_activation,
        output_activation,
    ) -> list[Layer]:
        """Build a list of layers."""
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            is_last = (i == len(layer_sizes) - 2)
            act = output_activation if is_last and output_activation else hidden_activation
            layers.append(Layer(in_dimm=in_dim, out_dimm=out_dim, activation_function=act))
        return layers

    def encode(self, x: NDArray) -> tuple[NDArray, NDArray]:
        """Encode input to (mean, log_var)."""
        mean = self._forward_layers(x, self.encoder_mean)
        log_var = self._forward_layers(x, self.encoder_logvar)
        return mean, log_var

    def reparameterize(self, mean: NDArray, log_var: NDArray) -> tuple[NDArray, NDArray]:
        """Reparameterization trick: z = mean + std * epsilon."""
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mean.shape)
        z = mean + std * epsilon
        return z, epsilon

    def decode(self, z: NDArray) -> NDArray:
        """Decode latent to output."""
        return self._forward_layers(z, self.decoder)

    def forward(self, x: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Full VAE forward: returns (reconstruction, mean, log_var, z, epsilon)."""
        mean, log_var = self.encode(x)
        z, epsilon = self.reparameterize(mean, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mean, log_var, z, epsilon

    def _forward_layers(self, x: NDArray, layers: list[Layer]) -> NDArray:
        """Forward pass through a list of layers."""
        out = x
        for layer in layers:
            out = layer.feed_forward(out)
        return out

    def _backward_layers(self, grad_output: NDArray, layers: list[Layer]) -> NDArray:
        """Backward pass through a list of layers. Returns gradient w.r.t. input."""
        grad_input = grad_output
        for layer in reversed(layers):
            grad_input, _, _ = layer.backward(grad_input)
        return grad_input

    def compute_loss(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        mean: NDArray,
        log_var: NDArray,
    ) -> tuple[float, float, float]:
        """
        Compute total VAE loss.

        Returns:
            (total_loss, recon_loss, kl_loss)
        """
        recon_loss = float(MeanSquaredError().compute(y_pred, y_true)) * y_true.size
        # KL divergence: D_KL(N(mean, exp(log_var)) || N(0, 1))
        kl_loss = -0.5 * np.sum(1 + log_var - mean**2 - np.exp(log_var))
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, float(kl_loss)

    def fit(
        self,
        X: NDArray,
        y: NDArray | None = None,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float | None = None,
        print_every: int = 100,
        validation_X: NDArray | None = None,
        validation_y: NDArray | None = None,
    ) -> list[dict]:
        """
        Train the VAE.

        Returns:
            List of dicts with 'loss', 'recon_loss', 'kl_loss', 'epoch', 'time'
        """
        if y is None:
            y = X
        if lr is None:
            lr = self.lr

        current_batch_size = batch_size or len(X)
        history = []

        # Initialize optimizers
        self.encoder_mean_opt_W = [SGD(lr=lr) for _ in self.encoder_mean]
        self.encoder_mean_opt_b = [SGD(lr=lr) for _ in self.encoder_mean]
        self.encoder_logvar_opt_W = [SGD(lr=lr) for _ in self.encoder_logvar]
        self.encoder_logvar_opt_b = [SGD(lr=lr) for _ in self.encoder_logvar]
        self.decoder_opt_W = [SGD(lr=lr) for _ in self.decoder]
        self.decoder_opt_b = [SGD(lr=lr) for _ in self.decoder]

        for epoch in range(1, epochs + 1):
            epoch_recon = 0.0
            epoch_kl = 0.0
            start_time = time.time()

            for start in range(0, len(X), current_batch_size):
                batch_X = X[start:start + current_batch_size]
                batch_y = y[start:start + current_batch_size]

                # Forward
                mean, log_var = self.encode(batch_X)
                z, epsilon = self.reparameterize(mean, log_var)
                y_pred = self.decode(z)

                # Loss
                total_loss, recon_loss, kl_loss = self.compute_loss(batch_y, y_pred, mean, log_var)
                epoch_recon += recon_loss
                epoch_kl += kl_loss

                # KL gradient w.r.t. log_var: -0.5 * (1 + log_var - mean^2 - exp(log_var))
                kl_grad_mean = -0.5 * (-2 * mean)  # d(KL)/d(mean)
                kl_grad_logvar = -0.5 * (1 - np.exp(log_var))  # d(KL)/d(log_var)

                # Gradient w.r.t. z from reconstruction loss
                recon_grad = MeanSquaredError().gradient(y_pred, batch_y)
                # Gradient w.r.t. z from KL (via mean and log_var)
                # z = mean + exp(log_var/2) * epsilon
                # dz/d(mean) = 1 + exp(log_var/2) * epsilon / exp(log_var/2) = ... 
                # Actually simpler: use chain rule through mean and log_var

                # Backward through decoder (manual, collect grads in reverse order)
                dec_grad = recon_grad
                decoder_grads = []
                for i in range(len(self.decoder) - 1, -1, -1):
                    layer = self.decoder[i]
                    dec_grad, grad_W, grad_b = layer.backward(dec_grad)
                    decoder_grads.append((grad_W, grad_b))
                decoder_grads.reverse()

                # Gradient at latent space (z) from reconstruction
                z_grad = dec_grad  # dL/dz from reconstruction

                # Backward through encoder mean (dL/d(mean) = z_grad * dz/d(mean) + KL_grad)
                # z = mean + exp(log_var/2) * epsilon, so dz/d(mean) = exp(log_var/2)
                grad_mean = z_grad * np.exp(0.5 * log_var) + self.kl_weight * kl_grad_mean
                enc_grad = grad_mean
                encoder_mean_grads = []
                for i in range(len(self.encoder_mean) - 1, -1, -1):
                    layer = self.encoder_mean[i]
                    enc_grad, grad_W, grad_b = layer.backward(enc_grad)
                    encoder_mean_grads.append((grad_W, grad_b))
                encoder_mean_grads.reverse()

                # Backward through encoder log_var
                # dz/d(log_var) = exp(log_var/2) * epsilon * 0.5
                grad_logvar = z_grad * (np.exp(0.5 * log_var) * epsilon * 0.5) + self.kl_weight * kl_grad_logvar
                enc_logvar_grad = grad_logvar
                encoder_logvar_grads = []
                for i in range(len(self.encoder_logvar) - 1, -1, -1):
                    layer = self.encoder_logvar[i]
                    enc_logvar_grad, grad_W, grad_b = layer.backward(enc_logvar_grad)
                    encoder_logvar_grads.append((grad_W, grad_b))
                encoder_logvar_grads.reverse()

                # Update decoder weights (grads in forward order)
                for i, layer in enumerate(self.decoder):
                    grad_W, grad_b = decoder_grads[i]
                    layer.W = self.decoder_opt_W[i].update(layer.W, grad_W)
                    layer.b = self.decoder_opt_b[i].update(layer.b, grad_b)

                # Update encoder mean weights
                for i, layer in enumerate(self.encoder_mean):
                    grad_W, grad_b = encoder_mean_grads[i]
                    layer.W = self.encoder_mean_opt_W[i].update(layer.W, grad_W)
                    layer.b = self.encoder_mean_opt_b[i].update(layer.b, grad_b)

                # Update encoder log_var weights
                for i, layer in enumerate(self.encoder_logvar):
                    grad_W, grad_b = encoder_logvar_grads[i]
                    layer.W = self.encoder_logvar_opt_W[i].update(layer.W, grad_W)
                    layer.b = self.encoder_logvar_opt_b[i].update(layer.b, grad_b)

            avg_recon = epoch_recon / len(X)
            avg_kl = epoch_kl / len(X)
            total = avg_recon + self.kl_weight * avg_kl
            elapsed = time.time() - start_time

            entry = {
                "epoch": epoch,
                "loss": total,
                "recon_loss": avg_recon,
                "kl_loss": avg_kl,
                "time": elapsed,
            }
            history.append(entry)

            msg = (
                f"VAE      | Epoch {epoch:4d}/{epochs} | "
                f"Loss: {total:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}) | "
                f"Time: {elapsed:.2f}s"
            )
            if validation_X is not None:
                val_total, val_recon, val_kl = self._evaluate_batch(validation_X, validation_y)
                msg += f" | Val: {val_total:.4f}"
            print(msg)

        return history

    def _evaluate_batch(self, X: NDArray, y: NDArray) -> tuple[float, float, float]:
        """Evaluate loss on a batch (no sampling noise)."""
        mean, log_var = self.encode(X)
        # Use mean directly (no stochastic sampling for validation)
        y_pred = self.decode(mean)
        total, recon, kl = self.compute_loss(y, y_pred, mean, log_var)
        return total, recon, kl

    def generate(self, n_samples: int = 10, seed: int | None = None) -> NDArray:
        """Generate new samples from the learned distribution."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        z = rng.standard_normal((n_samples, self.latent_dim))
        return self.decode(z)

    def get_latent_representation(self, X: NDArray) -> NDArray:
        """Get mean latent representation (for visualization)."""
        mean, _ = self.encode(X)
        return mean
