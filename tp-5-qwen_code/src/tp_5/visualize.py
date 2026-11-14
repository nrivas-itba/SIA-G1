"""
Visualization utilities for TP5 autoencoder results.

Generates all required graphics:
- Latent space scatter plot (TP5 1a3)
- Reconstruction examples (TP5 1a4)
- Loss curves
- Denoising results at different noise levels
- VAE latent space and generated samples
- Residual vs standard autoencoder comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_latent_space(
    latent: np.ndarray,
    labels: np.ndarray,
    title: str = "Latent Space (2D)",
    output_path: str | None = None,
    plot_type: str = "plotly",
) -> go.Figure:
    """Plot 2D latent space colored by class (TP5 1a3)."""
    fig = go.Figure()

    for label in sorted(np.unique(labels)):
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=latent[mask, 0],
            y=latent[mask, 1],
            mode="markers",
            name=str(int(label)),
            marker=dict(size=6, opacity=0.7),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="z₁",
        yaxis_title="z₂",
        template="plotly_white",
        width=700,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    )

    if output_path:
        fig.write_image(str(output_path), width=700, height=600, scale=2)
        print(f"  Saved latent space plot to {output_path}")

    return fig


def plot_reconstruction_examples(
    X_original: np.ndarray,
    X_reconstructed: np.ndarray,
    labels: np.ndarray,
    n_examples: int = 20,
    title: str = "Reconstruction Examples",
    output_path: str | None = None,
) -> go.Figure:
    """Show original vs reconstructed images (TP5 1a4)."""
    fig = make_subplots(
        rows=n_examples // 5 + 1,
        cols=2,
        subplot_titles=["Original", "Reconstructed"] * (n_examples // 2 + 1),
        vertical_spacing=0.08,
        horizontal_spacing=0.15,
    )

    # Select diverse examples (different labels)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_original), size=min(n_examples, len(X_original)), replace=False)

    for i, idx in enumerate(indices):
        original = X_original[idx].reshape(5, 7)
        reconstructed = X_reconstructed[idx].reshape(5, 7)

        row = i // 2 + 1
        col = 1 if i % 2 == 0 else 2

        fig.add_trace(
            go.Heatmap(
                z=original,
                colorscale="Gray",
                zmid=0.5,
                showscale=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=n_examples * 40,
    )

    if output_path:
        fig.write_image(str(output_path), width=800, height=n_examples * 40, scale=2)
        print(f"  Saved reconstruction examples to {output_path}")

    return fig


def plot_loss_curves(
    history: list,
    title: str = "Training Loss",
    output_path: str | None = None,
    y_label: str = "Loss",
) -> go.Figure:
    """Plot training and validation loss curves."""
    epochs = [h["epoch"] if isinstance(h, dict) else h for h in history]

    fig = go.Figure()

    if isinstance(history[0], dict):
        losses = [h["loss"] for h in history]
        recon_losses = [h.get("recon_loss", h["loss"]) for h in history]
        kl_losses = [h.get("kl_loss", 0) for h in history]

        fig.add_trace(go.Scatter(x=epochs, y=recon_losses, mode="lines", name="Reconstruction"))
        fig.add_trace(go.Scatter(x=epochs, y=kl_losses, mode="lines", name="KL Divergence"))
        fig.add_trace(go.Scatter(x=epochs, y=losses, mode="lines", name="Total"))
        y_label = "Loss"
    else:
        fig.add_trace(go.Scatter(x=epochs, y=history, mode="lines", name="Loss"))
        y_label = "Loss"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        template="plotly_white",
        width=800,
        height=500,
    )

    if output_path:
        fig.write_image(str(output_path), width=800, height=500, scale=2)
        print(f"  Saved loss curve to {output_path}")

    return fig


def plot_denoising_results(
    results: dict,
    output_path: str | None = None,
) -> go.Figure:
    """Plot denoising capacity at different noise levels."""
    noise_rates = sorted(results.keys())
    losses = [results[r]["reconstruction_loss"] for r in noise_rates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=noise_rates,
        y=losses,
        mode="markers+lines",
        name="Reconstruction Loss",
        marker=dict(size=10),
    ))

    fig.update_layout(
        title="Denoising Autoencoder: Loss vs Noise Level",
        xaxis_title="Noise Rate (%)",
        yaxis_title="Reconstruction Loss (MSE)",
        template="plotly_white",
        width=700,
        height=500,
    )

    if output_path:
        fig.write_image(str(output_path), width=700, height=500, scale=2)
        print(f"  Saved denoising results to {output_path}")

    return fig


def plot_vae_latent_space(
    latent: np.ndarray,
    labels: np.ndarray,
    title: str = "VAE Latent Space",
    output_path: str | None = None,
) -> go.Figure:
    """Plot VAE latent space."""
    fig = go.Figure()

    for label in sorted(np.unique(labels)):
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=latent[mask, 0],
            y=latent[mask, 1],
            mode="markers",
            name=str(int(label)),
            marker=dict(size=8, opacity=0.6),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="z₁",
        yaxis_title="z₂",
        template="plotly_white",
        width=700,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    )

    if output_path:
        fig.write_image(str(output_path), width=700, height=600, scale=2)
        print(f"  Saved VAE latent space to {output_path}")

    return fig


def plot_vae_reconstruction(
    X_original: np.ndarray,
    X_reconstructed: np.ndarray,
    n_examples: int = 10,
    title: str = "VAE Reconstruction Examples",
    output_path: str | None = None,
) -> go.Figure:
    """Show VAE reconstruction examples."""
    n_rows = (n_examples // 2) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=["Original", "Reconstructed"] * n_rows,
        vertical_spacing=0.08,
        horizontal_spacing=0.15,
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_original), size=min(n_examples, len(X_original)), replace=False)

    for i, idx in enumerate(indices):
        original = X_original[idx].reshape(5, 7)
        reconstructed = X_reconstructed[idx].reshape(5, 7)
        row = i // 2 + 1
        col = 1 if i % 2 == 0 else 2

        fig.add_trace(go.Heatmap(z=original, colorscale="Gray", zmid=0.5, showscale=(i == 0)), row=row, col=col)
        fig.add_trace(go.Heatmap(z=reconstructed, colorscale="Gray", zmid=0.5, showscale=(i == 0)), row=row, col=col + 1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=n_examples * 40,
    )

    if output_path:
        fig.write_image(str(output_path), width=800, height=n_examples * 40, scale=2)
        print(f"  Saved VAE reconstructions to {output_path}")

    return fig


def plot_generated_samples(
    samples: np.ndarray,
    labels: np.ndarray | None = None,
    n_examples: int = 20,
    title: str = "Generated VAE Samples",
    output_path: str | None = None,
) -> go.Figure:
    """Show VAE generated samples."""
    n_rows = (n_examples // 2) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f"Sample {i+1}" for i in range(n_examples)],
        vertical_spacing=0.08,
        horizontal_spacing=0.15,
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(samples), size=min(n_examples, len(samples)), replace=False)

    for i, idx in enumerate(indices):
        sample = samples[idx].reshape(5, 7)
        row = i // 2 + 1
        col = 1 if i % 2 == 0 else 2
        fig.add_trace(
            go.Heatmap(z=sample, colorscale="Viridis", showscale=(i == 0)),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=n_examples * 40,
    )

    if output_path:
        fig.write_image(str(output_path), width=800, height=n_examples * 40, scale=2)
        print(f"  Saved generated samples to {output_path}")

    return fig


def plot_comparison(
    standard_history: list,
    residual_history: list,
    standard_latent: np.ndarray | None = None,
    residual_latent: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    output_dir: str | None = None,
) -> dict:
    """Compare standard vs residual autoencoder."""
    if output_dir is None:
        output_dir = "/project/tp-5/results"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Loss comparison
    fig = go.Figure()
    epochs_std = [h["epoch"] if isinstance(h, dict) else h for h in standard_history]
    losses_std = [h["loss"] if isinstance(h, dict) else h for h in standard_history]
    fig.add_trace(go.Scatter(x=epochs_std, y=losses_std, mode="lines", name="Standard AE"))

    epochs_res = [h["epoch"] if isinstance(h, dict) else h for h in residual_history]
    losses_res = [h["loss"] if isinstance(h, dict) else h for h in residual_history]
    fig.add_trace(go.Scatter(x=epochs_res, y=losses_res, mode="lines", name="Residual AE"))

    fig.update_layout(
        title="Standard AE vs Residual AE: Loss Comparison",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        width=800,
        height=500,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    )
    fig.write_image(str(out / "comparison_loss.png"), width=800, height=500, scale=2)
    print(f"  Saved comparison loss to {out / 'comparison_loss.png'}")

    # Latent space comparison side-by-side
    if standard_latent is not None and residual_latent is not None and labels is not None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Standard AE Latent Space", "Residual AE Latent Space"],
        )

        for label in sorted(np.unique(labels)):
            mask = labels == label
            fig.add_trace(
                go.Scatter(x=standard_latent[mask, 0], y=standard_latent[mask, 1],
                           mode="markers", name=str(int(label)) if label == sorted(np.unique(labels))[0] else None,
                           marker=dict(size=6, opacity=0.7, color="blue")),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=residual_latent[mask, 0], y=residual_latent[mask, 1],
                           mode="markers", name=str(int(label)) if label == sorted(np.unique(labels))[0] else None,
                           marker=dict(size=6, opacity=0.7, color="red")),
                row=1, col=2,
            )

        fig.update_layout(
            template="plotly_white",
            width=1200,
            height=500,
        )
        fig.write_image(str(out / "comparison_latent.png"), width=1200, height=500, scale=2)
        print(f"  Saved comparison latent to {out / 'comparison_latent.png'}")

    return {"standard_history": standard_history, "residual_history": residual_history}


def generate_all_visualizations(
    results_dir: str = "/project/tp-5/results",
):
    """Generate all visualization scripts and call individual plotters."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating all visualizations")
    print(f"{'='*60}")

    # Will be called with actual data after training
    return out
