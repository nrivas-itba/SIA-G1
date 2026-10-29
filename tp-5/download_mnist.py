#!/usr/bin/env python3
"""Download MNIST from alternative sources or use TP-3 digit data."""
import os
import sys

DATA_DIR = "data/mnist"
os.makedirs(DATA_DIR, exist_ok=True)

# Alternative MNIST URLs (GitHub mirrors that are reliable)
ALTERNATIVE_URLS = [
    "https://raw.githubusercontent.com/alexlee-gk/deep-learning-tensorflow/master/data/mnist/train-images-idx3-ubyte.gz",
    "http://s3.amazonaws.com/img-datasets/mnist.tar.gz",
]

MNIST_URL_BASE = "http://yann.lecun.com/exdb/mnist"
files = {
    "train-images-idx3-ubyte.gz": f"{MNIST_URL_BASE}/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": f"{MNIST_URL_BASE}/train-labels-idx1-ubyte.gz",
}

import urllib.request

downloaded = False
for filename, url in files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Trying to download {filename}...")
        for try_url in [url] + ALTERNATIVE_URLS:
            try:
                urllib.request.urlretrieve(try_url, filepath)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  Done: {size_mb:.1f} MB")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed from {try_url}: {e}")
    else:
        print(f"{filename} already exists ({os.path.getsize(filepath)} bytes)")

if downloaded:
    print("Download complete!")
else:
    npz_path = os.path.join(DATA_DIR, "train.npz")
    if os.path.exists(npz_path):
        print(f"MNIST numpy data already present at {npz_path}")
    else:
        print("No MNIST data found. Please place raw .gz files or train.npz in data/mnist/")
