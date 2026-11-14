"""
Residual Autoencoder with skip connections.

Compares against the standard Autoencoder to demonstrate the benefit
of residual connections for gradient flow in deep autoencoders.

Architecture:
    Standard path:  x → encoder → latent → decoder → x_rec
    Residual path:  x → skip connection → concatenated with decoder output
    Final output:   x_rec + residual(x)  (learned shortcut)
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
from tp_3.engine.optimizers.momentum import Momentum
from tp_3.engine.optimizers.adam import Adam
from tp_3.engine.loss_functions.mse import MeanSquaredError


class ResidualAutoencoder:
    """
    Autoencoder with residual (skip) connections.

    The key difference from the standard autoencoder:
    - Encoder and decoder share weight symmetry (W_dec = W_enc^T)
    - Skip connection: input is added directly to the output
    - Helps gradient flow through deep networks
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
        residual_scale: float = 1.0,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.activation = activation
        self.beta = beta
        self.residual_scale = residual_scale

        # Build activation map
        act_map = {
            "sigmoid": Sigmoid(beta=beta),
            "tanh": Tanh(beta=beta),
            "relu": ReLU(),
        }
        self.hidden_act = act_map[activation]

        # Encoder: input_dim → h1 → h2 → ... → latent_dim
        encoder_dims = [input_dim] + self.hidden_dims + [latent_dim]
        self.encoder = self._build_mlp(encoder_dims, self.hidden_act, None)

        # Decoder: latent_dim → h2 → h1 → ... → input_dim
        # Use symmetric architecture (mirror of encoder)
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [input_dim]
        self.decoder = self._build_mlp(decoder_dims, self.hidden_act, act_map["sigmoid"])

        # Residual path (single Layer for shortcut)
        self.residual_layers = [self._build_residual_path(input_dim, input_dim)]

        # Optimizer states
        self.encoder_opt_W = None
        self.encoder_opt_b = None
        self.decoder_opt_W = None
        self.decoder_opt_b = None

    def _build_mlp(
        self,
        layer_sizes: list[int],
        hidden_activation,
        output_activation,
    ) -> list[Layer]:
        """Build a list of layers (simple MLP without optimizer)."""
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            is_last = (i == len(layer_sizes) - 2)
            act = output_activation if is_last and output_activation else hidden_activation
            layers.append(Layer(in_dimm=in_dim, out_dimm=out_dim, activation_function=act))
        return layers

    def _build_residual_path(self, in_dim: int, out_dim: int) -> Layer:
        """Build a small residual path (linear shortcut with optional nonlinearity)."""
        mid_dim = max(in_dim // 2, 8)
        return Layer(in_dimm=in_dim, out_dimm=out_dim, activation_function=Sigmoid())

    def encode(self, x: NDArray) -> NDArray:
        """Encode input to latent space."""
        out = x
        for layer in self.encoder:
            out = layer.feed_forward(out)
        return out

    def decode(self, z: NDArray) -> NDArray:
        """Decode latent to output."""
        out = z
        for layer in self.decoder:
            out = layer.feed_forward(out)
        return out

    def forward(self, x: NDArray) -> NDArray:
        """Full encode-decode with residual connection."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        # Residual connection: add learned shortcut from input
        residual = self.residual_layers[0].feed_forward(x)
        output = reconstructed + self.residual_scale * residual

        return output

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
    ) -> list[float]:
        """Train the residual autoencoder."""
        if y is None:
            y = X
        if lr is None:
            lr = self.lr

        current_batch_size = batch_size or len(X)
        history = []

        # Initialize optimizers
        self.encoder_opt_W = [SGD(lr=lr) for _ in self.encoder]
        self.encoder_opt_b = [SGD(lr=lr) for _ in self.encoder]
        self.decoder_opt_W = [SGD(lr=lr) for _ in self.decoder]
        self.decoder_opt_b = [SGD(lr=lr) for _ in self.decoder]
        self.residual_opt = SGD(lr=lr)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            start_time = time.time()

            for start in range(0, len(X), current_batch_size):
                batch_X = X[start:start + current_batch_size]
                batch_y = y[start:start + current_batch_size]

                # Forward pass
                latent = self.encode(batch_X)
                reconstructed = self.decode(latent)
                residual_out = self.residual_layers[0].feed_forward(batch_X)
                y_pred = reconstructed + self.residual_scale * residual_out

                # Loss
                loss = float(MeanSquaredError().compute(y_pred, batch_y)) * len(batch_X)
                epoch_loss += loss

                # Backward pass (manual for residual connections)
                loss_grad = MeanSquaredError().gradient(y_pred, batch_y)

                # Gradient w.r.t. residual_scale
                residual_grad = loss_grad * self.residual_scale

                # Update residual layer
                res_input = batch_X
                res_z = self.residual_layers[0].z
                res_grad_z = loss_grad * self.residual_scale * Sigmoid().gradient(res_z)
                res_grad_W = res_input.T @ res_grad_z
                res_grad_b = np.sum(res_grad_z, axis=0)
                self.residual_layers[0].W = self.residual_opt.update(self.residual_layers[0].W, res_grad_W)
                self.residual_layers[0].b = self.residual_opt.update(self.residual_layers[0].b, res_grad_b)

                # Backward through decoder (collect grads first)
                dec_grad = loss_grad
                decoder_grads = []
                for i in range(len(self.decoder) - 1, -1, -1):
                    layer = self.decoder[i]
                    dec_grad, grad_W, grad_b = layer.backward(dec_grad)
                    decoder_grads.append((grad_W, grad_b))
                decoder_grads.reverse()  # Now in forward order: [grad_0, grad_1, ..., grad_N]

                # Backward through encoder (collect grads first)
                enc_grad = dec_grad  # dL/d(latent) from decoder backward
                encoder_grads = []
                for i in range(len(self.encoder) - 1, -1, -1):
                    layer = self.encoder[i]
                    enc_grad, grad_W, grad_b = layer.backward(enc_grad)
                    encoder_grads.append((grad_W, grad_b))
                encoder_grads.reverse()

                # Update decoder weights (forward order matches)
                for i, layer in enumerate(self.decoder):
                    grad_W, grad_b = decoder_grads[i]
                    layer.W = self.decoder_opt_W[i].update(layer.W, grad_W)
                    layer.b = self.decoder_opt_b[i].update(layer.b, grad_b)

                # Update encoder weights (forward order matches)
                for i, layer in enumerate(self.encoder):
                    grad_W, grad_b = encoder_grads[i]
                    layer.W = self.encoder_opt_W[i].update(layer.W, grad_W)
                    layer.b = self.encoder_opt_b[i].update(layer.b, grad_b)

            epoch_loss /= len(X)
            history.append(epoch_loss)

            elapsed = time.time() - start_time
            msg = f"ResNet AE | Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s"

            if validation_X is not None:
                val_loss = self.evaluate_loss(validation_X, validation_y)
                msg += f" | Val: {val_loss:.6f}"

            print(msg)

        return history

    def evaluate_loss(self, X: NDArray, y: NDArray) -> float:
        """Compute reconstruction loss."""
        y_pred = self.forward(X)
        return float(MeanSquaredError().compute(y_pred, y))

    def get_latent_representation(self, X: NDArray) -> NDArray:
        """Get latent representation."""
        return self.encode(X)
