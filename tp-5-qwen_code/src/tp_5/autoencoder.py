"""
Autoencoder built on TP3 engine (Layer, MLP, optimizers, loss functions).

Extends the TP3 MLP to create encoder-decoder architectures suitable
for reconstructing 5x7 binary images into a 2D latent space.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Reuse TP3 engine
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tp-3"))
from tp_3.engine.layer import Layer
from tp_3.engine.activation_functions.tanh import Tanh
from tp_3.engine.activation_functions.sigmoid import Sigmoid
from tp_3.engine.activation_functions.relu import ReLU
from tp_3.engine.optimizers.sgd import SGD
from tp_3.engine.optimizers.momentum import Momentum
from tp_3.engine.optimizers.adam import Adam
from tp_3.engine.loss_functions.mse import MeanSquaredError
from tp_3.engine.models.mlp import MLP


class Autoencoder(MLP):
    """
    Autoencoder extending TP3's MLP.

    Architecture:
        Encoder: input → hidden_layers → latent (2D)
        Decoder: latent → hidden_layers → output (same dim as input)

    For binary reconstruction: sigmoid activation in final decoder layer.
    """

    def __init__(
        self,
        input_dim: int = 35,  # 5x7
        latent_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
        output_activation: str = "sigmoid",
        lr: float = 0.01,
        optimizer: str = "adam",
        beta: float = 1.0,
    ):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Input/output dimension (e.g., 35 for 5x7)
            latent_dim: Latent space dimension (must be 2 for TP5)
            hidden_dims: Encoder hidden layer sizes (decoder mirrors)
            activation: Hidden activation ('sigmoid', 'tanh', 'relu')
            output_activation: Final layer activation ('sigmoid' for binary)
            lr: Learning rate
            optimizer: Optimizer type ('sgd', 'momentum', 'adam')
            beta: Activation beta coefficient
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.activation = activation
        self.beta = beta

        # Build activation map
        act_map = {
            "sigmoid": Sigmoid(beta=beta),
            "tanh": Tanh(beta=beta),
            "relu": ReLU(),
        }
        self.hidden_act = act_map[activation]

        # Build full architecture
        encoder_dims = [input_dim] + self.hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [input_dim]

        # Create encoder model
        encoder = self._build_model(
            layer_sizes=encoder_dims,
            hidden_activation=self.hidden_act,
            output_activation=None,  # Linear output for latent
        )

        # Create decoder model
        output_act = act_map.get(output_activation)
        decoder = self._build_model(
            layer_sizes=decoder_dims,
            hidden_activation=self.hidden_act,
            output_activation=output_act,
        )

        self.encoder = encoder
        self.decoder = decoder

    def _build_model(
        self,
        layer_sizes: list[int],
        hidden_activation,
        output_activation,
    ) -> MLP:
        """Build an MLP model."""
        act_map = {
            "sgd": SGD,
            "momentum": lambda lr: Momentum(lr=lr, momentum=0.9),
            "adam": Adam,
        }

        loss = MeanSquaredError()

        model = MLP(
            layer_sizes=layer_sizes,
            input_activation=hidden_activation,
            output_activation=output_activation,
            loss_function=loss,
            optimizer=SGD,
        )
        return model

    def encode(self, x: NDArray) -> NDArray:
        """Encode input to latent space."""
        return self._forward_layers(x, self.encoder.layers)

    def decode(self, z: NDArray) -> NDArray:
        """Decode latent representation to output."""
        return self._forward_layers(z, self.decoder.layers)

    def forward(self, x: NDArray) -> NDArray:
        """Full encode-then-decode pass."""
        latent = self.encode(x)
        return self.decode(latent)

    def _forward_layers(self, x: NDArray, layers: list) -> NDArray:
        """Forward pass through a list of layers (TP3 Layer objects)."""
        out = x
        for layer in layers:
            out = layer.feed_forward(out)
        return out

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
        callback_epochs: set[int] | None = None,
    ) -> list[float]:
        """
        Train the autoencoder (encoder + decoder jointly).

        Args:
            X: Input data (n_samples, input_dim)
            y: Target data (same as X for reconstruction)
            batch_size: Mini-batch size
            epochs: Number of training epochs
            lr: Learning rate override
            print_every: Print loss every N epochs
            validation_X, validation_y: Validation data
            callback_epochs: Epochs to trigger callbacks

        Returns:
            Training loss history
        """
        if y is None:
            y = X  # Self-supervised: target = input
        if lr is None:
            lr = self.lr

        history = []
        current_batch_size = batch_size or len(X)
        is_full_batch = (current_batch_size == len(X))

        # Initialize optimizer states for all layers
        encoder_opt_W = [SGD(lr=lr) for _ in self.encoder.layers]
        encoder_opt_b = [SGD(lr=lr) for _ in self.encoder.layers]
        decoder_opt_W = [SGD(lr=lr) for _ in self.decoder.layers]
        decoder_opt_b = [SGD(lr=lr) for _ in self.decoder.layers]

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            start_time = time.time()

            for start in range(0, len(X), current_batch_size):
                batch_X = X[start:start + current_batch_size]
                batch_y = y[start:start + current_batch_size]

                # Forward: encode
                latent = self._forward_layers(batch_X, self.encoder.layers)
                # Forward: decode
                y_pred = self._forward_layers(latent, self.decoder.layers)

                # Loss
                mse = MeanSquaredError()
                loss = float(mse.compute(y_pred, batch_y)) * len(batch_X)
                epoch_loss += loss

                # Backward: decoder (manual, collect grads in reverse order)
                loss_grad = mse.gradient(y_pred, batch_y)
                grad = loss_grad
                decoder_grads = []  # [grad_layer_N, grad_layer_N-1, ..., grad_layer_0]
                for i in range(len(self.decoder.layers) - 1, -1, -1):
                    layer = self.decoder.layers[i]
                    grad, grad_W, grad_b = layer.backward(grad)
                    decoder_grads.append((grad_W, grad_b))
                decoder_grads.reverse()  # Now in forward order: [grad_0, grad_1, ..., grad_N]

                # Gradient at encoder output (grad after decoder backward)
                latent_grad = grad  # This is dL/d(latent)

                # Backward: encoder (manual)
                encoder_grads = []
                for i in range(len(self.encoder.layers) - 1, -1, -1):
                    layer = self.encoder.layers[i]
                    grad, grad_W, grad_b = layer.backward(grad)
                    encoder_grads.append((grad_W, grad_b))
                encoder_grads.reverse()

                # Update decoder weights
                for i, layer in enumerate(self.decoder.layers):
                    grad_W, grad_b = decoder_grads[i]
                    layer.W = decoder_opt_W[i].update(layer.W, grad_W)
                    layer.b = decoder_opt_b[i].update(layer.b, grad_b)

                # Update encoder weights
                for i, layer in enumerate(self.encoder.layers):
                    grad_W, grad_b = encoder_grads[i]
                    layer.W = encoder_opt_W[i].update(layer.W, grad_W)
                    layer.b = encoder_opt_b[i].update(layer.b, grad_b)

            epoch_loss /= len(X)
            history.append(epoch_loss)

            elapsed = time.time() - start_time
            msg = f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s"

            if validation_X is not None:
                val_loss = self.evaluate_loss(validation_X, validation_y)
                msg += f" | Val: {val_loss:.6f}"

            print(msg)

            if callback_epochs and epoch in callback_epochs:
                yield epoch, self, epoch_loss

        return history

    def evaluate_loss(self, X: NDArray, y: NDArray) -> float:
        """Compute reconstruction loss without training."""
        y_pred = self.forward(X)
        return float(MeanSquaredError().compute(y_pred, y))

    def get_latent_representation(self, X: NDArray) -> NDArray:
        """Get the 2D latent representation of input data."""
        return self.encode(X)


def build_autoencoder_from_config(config: dict) -> Autoencoder:
    """Build autoencoder from config dict (supports TOML parsing)."""
    return Autoencoder(
        input_dim=config.get("input_dim", 35),
        latent_dim=config.get("latent_dim", 2),
        hidden_dims=config.get("hidden_dims", [64, 32]),
        activation=config.get("activation", "tanh"),
        output_activation=config.get("output_activation", "sigmoid"),
        lr=config.get("learning_rate", 0.01),
        optimizer=config.get("optimizer", "adam"),
        beta=config.get("beta", 1.0),
    )
