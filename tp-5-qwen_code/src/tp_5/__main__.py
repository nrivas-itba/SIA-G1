"""CLI entry point for TP5."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_all import main as run_all


def main():
    parser = argparse.ArgumentParser(description="TP5: Deep Learning - Autoencoders and VAEs")
    parser.add_argument("--experiment", choices=["all", "autoencoder", "residual", "denoising", "vae", "profile"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of hyperparameter optimization trials")
    args = parser.parse_args()

    if args.experiment == "all":
        run_all()
    elif args.experiment == "autoencoder":
        from tp_5.train_all import prepare_data, run_standard_autoencoder
        X, labels = prepare_data()
        run_standard_autoencoder(X, labels)
    elif args.experiment == "residual":
        from tp_5.train_all import prepare_data, run_residual_autoencoder
        X, labels = prepare_data()
        run_residual_autoencoder(X, labels)
    elif args.experiment == "denoising":
        from tp_5.train_all import prepare_data, run_denoising_autoencoder
        X, labels = prepare_data()
        run_denoising_autoencoder(X, labels)
    elif args.experiment == "vae":
        from tp_5.train_all import prepare_data, run_vae
        X, labels = prepare_data()
        run_vae(X, labels)
    elif args.experiment == "profile":
        from tp_5.train_all import prepare_data, run_standard_autoencoder, run_py_spy_profiling
        X, labels = prepare_data()
        ae, _ = run_standard_autoencoder(X, labels)
        run_py_spy_profiling(X, ae)


if __name__ == "__main__":
    main()
