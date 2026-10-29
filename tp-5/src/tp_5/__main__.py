"""Main training script for TP-5 Autoencoder."""

import sys
import json
import pickle
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tp_5.engine.models.autoencoder import Autoencoder
from tp_5.engine.activation_functions.relu import ReLU
from tp_5.engine.optimizers.adam import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
RESULTS_DIR = PROJECT_ROOT / "results"

LATENT_DIM = 32
HIDDEN_DIMS = [128, 64]
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PRINT_EVERY = 5


def load_digit_data():
    """Load handwritten digit data from CSV files or numpy npz format."""
    # Check for numpy npz format first (fastest)
    npz_path = DATA_DIR / "train.npz"
    if npz_path.exists():
        print(f"[INFO] Loading MNIST data from: {npz_path}")
        data = np.load(npz_path)
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.float64)
        return X, y

    # Check for pre-converted numpy file from CSV conversion
    csv_npz = DATA_DIR / "from_csv.npz"
    if csv_npz.exists():
        print(f"[INFO] Loading digit data from converted CSV: {csv_npz}")
        data = np.load(csv_npz)
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.float64)
        return X, y

    # Fallback to CSV loading (slower, for legacy TP-3 compatibility)
    candidate_paths = [
        DATA_DIR / "digits.csv",
        PROJECT_ROOT.parent / "tp-3" / "data" / "more_digits.csv",
        PROJECT_ROOT.parent / "tp-3" / "data" / "many_digits.csv",
        PROJECT_ROOT.parent / "tp-3" / "data" / "digits.csv",
    ]

    csv_path = None
    for p in candidate_paths:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "No digit data found. Expected it at one of:\n" +
            "\n".join(str(p) for p in candidate_paths)
        )

    print(f"[INFO] Loading digit data from CSV (first converting to numpy): {csv_path}")
    
    # Parse CSV format: label,"[val1, val2, ...]"
    labels = []
    images = []
    count = 0
    with open(csv_path, "r") as f:
        header = next(f)  # skip header line
        for line in f:
            if count >= 10000:
                break
            line = line.strip()
            if not line:
                continue
            # Find the comma that separates label from quoted image array
            first_comma = line.index(',')
            label = float(line[:first_comma].strip())
            # Extract the JSON array (between [ and ])
            img_json_start = line.index('[')
            img_json_end = line.rindex(']')
            img_data = json.loads(line[img_json_start:img_json_end+1])
            labels.append(label)
            images.append(img_data)
            count += 1

    X = np.array(images, dtype=np.float64) / 255.0
    y = np.array(labels, dtype=np.float64)
    
    # Save to numpy for future fast loading
    np.savez_compressed(csv_npz, X=X, y=y)
    print(f"[INFO] Converted {len(X)} samples to {csv_npz}")
    return X, y



def select_sample_indices(X_raw_labels, n_per_digit=3):
    """Select representative indices per digit (0-9)."""
    indices = []
    for d in range(10):
        digit_idx = np.where(np.round(X_raw_labels) == d)[0]
        step = max(len(digit_idx) // n_per_digit, 1)
        selected = digit_idx[::step][:n_per_digit]
        indices.extend(selected.tolist())
    return np.array(indices)


def run():
    """Main entry point for TP-5 Autoencoder training and visualization."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load digit data
    X_train_raw, y_train_labels = load_digit_data()

    # Step 2: Prepare data - flatten images and normalize
    n_samples = X_train_raw.shape[0]
    X_flat = X_train_raw.reshape(n_samples, -1)

    subset_size = min(500, len(X_flat))
    print(f"[INFO] Using subset of {subset_size} samples for training")
    X_train = X_flat[:subset_size]
    y_train_labels_subset = y_train_labels[:subset_size]  # Save labels BEFORE overwriting
    y_train = X_flat[:subset_size]  # Autoencoder: input = target

    print(f"[INFO] Data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Step 3: Train autoencoder
    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)

    input_dim = X_train.shape[1]
    output_dim = X_train.shape[1]

    model = Autoencoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, latent_dim=LATENT_DIM,
        output_dim=output_dim, activation=ReLU(),
    )

    arch_str = f"{input_dim} -> {' -> '.join(str(d) for d in HIDDEN_DIMS)} -> {LATENT_DIM} -> {' -> '.join(str(d) for d in reversed(HIDDEN_DIMS))} -> {output_dim}"
    print(f"[INFO] Architecture: {arch_str}")
    print(f"[INFO] Training with Adam, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")

    history = model.fit(
        X=X_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        lr=LEARNING_RATE, print_every=PRINT_EVERY,
        optimizer_factory=lambda: Adam(lr=LEARNING_RATE),
    )

    # Step 4: Get reconstructions and latent representations
    print("\n[INFO] Generating reconstructions...")
    sample_indices = select_sample_indices(y_train_labels_subset, n_per_digit=3)

    X_samples = X_flat[sample_indices]
    latent_representations = model.get_latent(X_samples).reshape(X_samples.shape[0], -1)
    reconstructions = model.reconstruct(X_samples)

    # Step 5: Generate HTML report
    output_html = RESULTS_DIR / "reconstruction_results.html"
    print(f"[INFO] Generating HTML report at {output_html}")
    
    # Save intermediate data before expensive HTML generation
    np.savez(RESULTS_DIR / "latent_representations.npz", X=latent_representations)
    np.savez(RESULTS_DIR / "reconstructions.npz", X=reconstructions)
    np.savez(RESULTS_DIR / "sample_indices.npz", X=sample_indices.astype(int))
    np.savez(RESULTS_DIR / "X_samples.npz", X=X_train_raw[:subset_size][sample_indices])
    _generate_html_report(history, X_train_raw[:subset_size], y_train_labels_subset,
                          sample_indices, latent_representations, reconstructions, output_html, input_dim=input_dim, output_dim=output_dim)

    # Step 6: Save model and results
    model_path = RESULTS_DIR / "autoencoder_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": model, "history": history}, f)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Final Loss: {history[-1]:.6f}")
    print(f"Results saved to: results/reconstruction_results.html")
    print(f"Model saved to: results/autoencoder_model.pkl")
    print(f"{'='*60}")

    return model, history, output_html




def _generate_html_report(history, X_train_raw, X_train_labels, sample_indices,
                           latent_representations, reconstructions, output_file, input_dim=None, output_dim=None):
    # PCA to 2D
    mean_lat = np.mean(latent_representations, axis=0)
    centered = latent_representations - mean_lat
    cov = np.dot(centered.T, centered) / len(centered)
    eigs_val, eigs_vec = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigs_val)[::-1]
    top2 = eigs_vec[:, sort_idx[:2]]
    latent_2d = np.dot(centered, top2)

    true_digits = np.round(X_train_labels).astype(int)
    digit_groups = {d: [] for d in range(10)}
    for i, idx in enumerate(sample_indices):
        digit_groups[true_digits[idx]].append((i, latent_2d[i]))

    def to_base64_png(arr, size=80):
        """Convert a 2D numpy array to base64 PNG image."""
        from PIL import Image
        import io, base64
        img = (arr * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG', quality=85)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # Generate image tags for orig/recon pairs once
    n_samples = len(sample_indices)
    orig_imgs_b64 = []
    recon_imgs_b64 = []
    for i in range(n_samples):
        orig_img = X_train_raw[sample_indices[i]].reshape(28, 28)
        recon_img = reconstructions[i].reshape(28, 28)
        orig_imgs_b64.append(to_base64_png(orig_img, size=80))
        recon_imgs_b64.append(to_base64_png(recon_img, size=80))

    loss_str = ", ".join(f"{v:.6f}" for v in history)
    labels_str = ", ".join(str(i+1) for i in range(len(history)))
    arch_parts = " -> ".join(str(d) for d in HIDDEN_DIMS)
    rev_arch = " -> ".join(str(d) for d in reversed(HIDDEN_DIMS))

    html_parts = []
    html_parts.append('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">')
    html_parts.append('<title>TP-5 Autoencoder Results</title>')
    html_parts.append('<style>')
    html_parts.append('body{font-family:Segoe UI,Tahoma,sans-serif;margin:20px;background:#f8f9fa;color:#333}')
    html_parts.append('h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px}h2{color:#2980b9;margin-top:30px}')
    html_parts.append('.info-box{background:#fff;border-left:4px solid #3498db;padding:15px;margin:15px 0;box-shadow:0 2px 4px rgba(0,0,0,0.1)}')
    html_parts.append('.chart-container{background:#fff;border-radius:8px;padding:20px;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);max-width:900px}')
    html_parts.append('.row{display:flex;gap:20px;align-items:center;margin:5px 0}.digit-label{font-weight:bold;width:80px;color:#2c3e50}')
    html_parts.append('</style><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script></head><body>')
    html_parts.append(f'<h1>TP-5 Autoencoder Results</h1>')
    if input_dim is None:
        input_dim = X_train_raw.shape[1]
    if output_dim is None:
        output_dim = X_train_raw.shape[1]
    html_parts.append('<div class="info-box"><strong>Architecture:</strong> ' + str(input_dim) + ' -> ' + arch_parts + ' -> ' + str(LATENT_DIM) + ' -> ' + rev_arch + ' -> ' + str(output_dim) + '<br>')
    html_parts.append('<strong>Optimizer:</strong> Adam (lr=' + str(LEARNING_RATE) + ') | <strong>Epochs:</strong> ' + str(EPOCHS) + ' | <strong>Batch size:</strong> ' + str(BATCH_SIZE) + '<br>')
    html_parts.append('<strong>Loss function:</strong> Mean Squared Error</div>')
    html_parts.append('<h2>1. Training Loss Curve</h2><div class="chart-container"><canvas id="lossChart"></canvas></div>')
    html_parts.append('<h2>2. Latent Space Visualization (PCA 2D)</h2><div class="chart-container"><canvas id="latentChart"></canvas></div>')
    html_parts.append('<h2>3. Original vs Reconstructed Images</h2><div class="chart-container">')

    for d in range(10):
        pts = digit_groups[d]
        if not pts:
            continue
        html_parts.append(f'<table><tr><th colspan="3">Digit {d}</th></tr>')
        # pts is a list of (index_in_reconstructions, latent_point) tuples
        for img_i, (recon_idx, _) in enumerate(pts):
            if img_i == 0:
                html_parts.append('<tr>')
            orig_img = X_train_raw[sample_indices[recon_idx]].reshape(28, 28)
            recon_img = reconstructions[recon_idx].reshape(28, 28)
            orig_b64 = to_base64_png(orig_img, size=80)
            recon_b64 = to_base64_png(recon_img, size=80)
            html_parts.append(f'<td><div class="row"><span class="digit-label">Orig</span><img src="{orig_b64}" width="80" height="80"/></div></td>')
            html_parts.append(f'<td><div class="row"><span class="digit-label">Recon</span><img src="{recon_b64}" width="80" height="80"/></div></td>')
            if img_i == len(pts) - 1:
                html_parts.append('</tr>\n')
        html_parts.append('</table>')

    html_parts.append('</div>\n<script>')
    html_parts.append('var LOSS_DATA=[' + loss_str + '];var LOSS_LABELS=[' + labels_str + '];')
    html_parts.append("var lc=document.getElementById('lossChart').getContext('2d');")
    html_parts.append("new Chart(lc,{type:'line',data:{labels:LOSS_LABELS,datasets:[{label:'Reconstruction Loss (MSE)',data:LOSS_DATA, borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,0.1)', borderWidth:2,fill:true,tension:0.3,pointRadius:3}]},options:{responsive:true,plugins:{title:{display:true,text:'Training Loss Over Epochs'}},scales:{x:{title:{display:true,text:'Epoch'}},y:{title:{display:true,text:'Loss (MSE)'}}}}});")

    colors = ["231,76,60","52,152,219","46,204,113","241,196,15","155,89,182","52,73,94","230,126,34","142,68,173","44,62,80","192,57,43"]
    html_parts.append("var slc=document.getElementById('latentChart').getContext('2d');")
    html_parts.append("var ds=[];")
    for d in range(10):
        pts = digit_groups[d]
        if not pts:
            continue
        data_str = ",".join(f"{{x:{pts[j][1][0]:.4f},y:{pts[j][1][1]:.4f}}}" for j in range(len(pts)))
        html_parts.append(f'ds.push({{label:"Digit {d}",data:[{data_str}],backgroundColor:"rgb({colors[d]})",pointRadius:3,pointHoverRadius:5}});')

    final_loss = history[-1]
    min_loss = min(history)
    min_epoch = history.index(min_loss) + 1
    html_parts.append("new Chart(slc,{type:'scatter',data:{datasets:ds},options:{responsive:true,plugins:{title:{display:true,text:'Latent Space (2D PCA)'},legend:{position:'bottom'}},scales:{x:{title:{display:true,text:'PC1'}},y:{title:{display:true,text:'PC2'}}}}});")
    html_parts.append(f"document.getElementById('lossChart').parentElement.insertAdjacentHTML('afterend','<div class=\\'info-box\\' style=\\'margin-top:10px;\\'><strong>Final Loss:</strong> {final_loss:.6f} | <strong>Min Loss:</strong> {min_loss:.6f} (epoch {min_epoch})</div>');")
    html_parts.append('</script><hr style="margin-top:40px;border-color:#ddd;"><p style="color:#999;text-align:center;">Generated by TP-5 Autoencoder</p></body></html>')

    with open(output_file, "w") as f:
        f.write("".join(html_parts))
    return output_file


if __name__ == "__main__":
    run()
