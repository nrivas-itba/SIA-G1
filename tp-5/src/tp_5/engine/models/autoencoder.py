"""Autoencoder model built on top of the TP-3 engine."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from .base import BaseModel
from ..layer import Layer
from ..activation_functions.base_activation import ActivationFunction
from ..activation_functions.relu import ReLU
from ..activation_functions.identity import Identity
from ..optimizers.base import Optimizer
from ..loss_functions.base import Loss
from ..loss_functions.mse import MeanSquaredError


class Autoencoder(BaseModel):
    """A simple autoencoder with configurable encoder and decoder architectures.

    Architecture: Input -> [Encoder Layers] -> Latent Vector -> [Decoder Layers] -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        output_dim: int | None = None,
        activation: ActivationFunction | None = None,
        loss_function: Loss | None = None,
    ):
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.activation = activation or ReLU()
        self.loss_fn = loss_function or MeanSquaredError()

        # Build encoder path
        self._build_encoder(input_dim, hidden_dims, latent_dim)
        # Build decoder path
        decoder_dims = [latent_dim] + list(reversed(hidden_dims)) + [output_dim]
        self._build_decoder(decoder_dims)

    def _build_encoder(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        self.encoder_layers: list[Layer] = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            is_last_encoder = (i == len(hidden_dims) - 1)
            act = Identity() if is_last_encoder else self.activation
            self.encoder_layers.append(Layer(in_dimm=prev_dim, out_dimm=h_dim, activation_function=act))
            prev_dim = h_dim
        if prev_dim != self.latent_dim:
            self.encoder_layers.append(
                Layer(in_dimm=prev_dim, out_dimm=self.latent_dim, activation_function=Identity())
            )

    def _build_decoder(self, decoder_dims: list[int]) -> None:
        self.decoder_layers: list[Layer] = []
        for i in range(len(decoder_dims) - 1):
            in_dim, out_dim = decoder_dims[i], decoder_dims[i + 1]
            is_last = (i == len(decoder_dims) - 2)
            act = Identity() if is_last else self.activation
            self.decoder_layers.append(Layer(in_dimm=in_dim, out_dimm=out_dim, activation_function=act))

    def encode(self, x: NDArray) -> NDArray:
        out = x
        for layer in self.encoder_layers:
            out = layer.feed_forward(out)
        return out

    def decode(self, z: NDArray) -> NDArray:
        out = z
        for layer in self.decoder_layers:
            out = layer.feed_forward(out)
        return out

    def forward(self, x: NDArray) -> NDArray:
        latent = self.encode(x)
        return self.decode(latent)

    def backward_encoder(self, grad_output: NDArray) -> list[tuple[NDArray, NDArray]]:
        grads = []
        grad_input = grad_output
        for layer in reversed(self.encoder_layers):
            grad_input, grad_W, grad_b = layer.backward(grad_input)
            grads.append((grad_W, grad_b))
        grads.reverse()
        return grads

    def backward_decoder(self, grad_output: NDArray) -> list[tuple[NDArray, NDArray]]:
        grads = []
        grad_input = grad_output
        for layer in reversed(self.decoder_layers):
            grad_input, grad_W, grad_b = layer.backward(grad_input)
            grads.append((grad_W, grad_b))
        grads.reverse()
        return grads
    def fit(
        self,
        X: NDArray,
        batch_size: int | None = None,
        epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        optimizer_factory: Callable[..., Optimizer] | None = None,
    ) -> list[float]:
        """Train the autoencoder on the given data."""
        if len(X) == 0:
            raise ValueError("Training data must contain at least one sample.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if optimizer_factory is None:
            from ..optimizers.adam import Adam
            optimizer_factory = lambda: Adam(lr=lr)

        optimizers_W_enc = [optimizer_factory() for _ in self.encoder_layers]
        optimizers_b_enc = [optimizer_factory() for _ in self.encoder_layers]
        optimizers_W_dec = [optimizer_factory() for _ in self.decoder_layers]
        optimizers_b_dec = [optimizer_factory() for _ in self.decoder_layers]

        current_batch_size = batch_size or len(X)
        is_full_batch = (current_batch_size == len(X))
        history: list[float] = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for start in range(0, len(X), current_batch_size):
                batch_X = X if is_full_batch else X[start : start + current_batch_size]
                y_pred = self.forward(batch_X)
                epoch_loss += float(self.loss_fn.compute(y_pred, batch_X)) * len(batch_X)

                loss_grad = self.loss_fn.gradient(y_pred, batch_X)
                dec_grads = self.backward_decoder(loss_grad)

                grad_at_latent = loss_grad.copy()
                for layer in reversed(self.decoder_layers):
                    grad_at_latent = np.matmul(grad_at_latent, layer.W.T)
                enc_grads = self.backward_encoder(grad_at_latent)

                for i, layer in enumerate(self.decoder_layers):
                    gW, gb = dec_grads[i]
                    layer.W = optimizers_W_dec[i].update(layer.W, gW)
                    layer.b = optimizers_b_dec[i].update(layer.b, gb)

                for i, layer in enumerate(self.encoder_layers):
                    gW, gb = enc_grads[i]
                    layer.W = optimizers_W_enc[i].update(layer.W, gW)
                    layer.b = optimizers_b_enc[i].update(layer.b, gb)

            epoch_loss /= len(X)
            history.append(epoch_loss)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | Reconstruction Loss (MSE): {epoch_loss:.6f}")

        return history

    def get_latent(self, x: NDArray) -> NDArray:
        """Get the latent representation for given inputs."""
        return self.encode(x)

    def reconstruct(self, x: NDArray) -> NDArray:
        """Reconstruct inputs through the autoencoder."""
        return self.forward(x)

