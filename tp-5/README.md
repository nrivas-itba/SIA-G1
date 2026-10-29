# TP-5: Autoencoder Implementation

## Overview

This project implements a simple Autoencoder built on top of a custom neural network engine (based on TP-3). The autoencoder learns to compress and reconstruct data by passing it through a bottleneck layer.

## Architecture

The autoencoder consists of two parts:
- **Encoder**: Compresses input data into a lower-dimensional representation
- **Decoder**: Reconstructs the original data from the compressed representation

## Components

### Engine (copied from TP-3)
- **Layers**: Dense/Linear layers with configurable activation functions
- **Activation Functions**: ReLU, Sigmoid, Tanh, Identity, Step
- **Loss Functions**: Mean Squared Error (MSE), Binary Cross Entropy (BCE)
- **Optimizers**: SGD, Momentum, RMSProp, Adam

### Autoencoder Model
- Builds on top of the MLP architecture
- Configurable encoder/decoder architecture
- Trained with MSE loss for reconstruction

## Usage

### Train the Autoencoder

```bash
# From the tp-5 directory:
uv run python -m tp_5
```

Or run directly:

```bash
uv run src/tp_5/__main__.py
```

### View Results

After training, open `results/reconstruction_results.html` in your browser to see:
- Original vs reconstructed images
- Loss curve during training
- Visualization of learned latent representations

## Dataset

The autoencoder is trained on the MNIST dataset (handwritten digits). The raw MNIST data files are downloaded automatically and cached in `data/mnist/`.

## File Structure

```
tp-5/
├── pyproject.toml
├── README.md
├── src/
│   └── tp_5/
│       ├── __init__.py
│       ├── __main__.py          # Main training script
│       └── engine/
│           ├── layer.py
│           ├── models/
│           │   ├── base.py      # BaseModel (ABC)
│           │   ├── mlp.py       # MLP classifier
│           │   └── autoencoder.py  # Autoencoder model
│           ├── optimizers/
│           ├── activation_functions/
│           └── loss_functions/
├── data/                        # Dataset files (MNIST)
└── results/                     # Output artifacts (HTML reports, saved models)
```

## Requirements

- Python 3.14+
- numpy
- matplotlib
- rust-matmul (for efficient matrix multiplication)

## Training Details

- **Loss**: Mean Squared Error (MSE) - appropriate for reconstruction tasks
- **Optimizer**: Adam (default)
- **Activation**: ReLU for hidden layers, Identity for output (reconstruction)
- **Data**: MNIST 28x28 = 784 pixels, normalized to [0, 1]
