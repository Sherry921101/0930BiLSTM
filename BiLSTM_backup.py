import argparse
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. Data loading -- VARIABLE length episodes
# ==========================================
def load_episode_files(file_dir, pattern, u_col, i_col, event_col, downsample_factor=1):
    """Each raw CSV file contains multiple events marked row-by-row via `event_col`.
    We group rows by that column to split each file into its individual events (episodes).
    """
    paths = sorted(glob.glob(os.path.join(file_dir, pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No files matched '{pattern}' in directory '{file_dir}'.\n"
            f"Please ensure your dataset files are placed in '{file_dir}' "
            f"or specify a custom path using --data_dir <path>."
        )

    U_list, I_list, episode_ids = [], [], []
    for p in paths:
        df = pd.read_csv(p)
        if event_col not in df.columns:
            raise KeyError(
                f"Column '{event_col}' not found in {p}. "
                f"Available columns: {list(df.columns)}"
            )
        if u_col not in df.columns or i_col not in df.columns:
            raise KeyError(
                f"Columns '{u_col}' or '{i_col}' not found in {p}. "
                f"Available columns: {list(df.columns)}"
            )

        file_tag = os.path.splitext(os.path.basename(p))[0]

        for event_no, group in df.groupby(event_col, sort=False):
            u = group[u_col].values.astype(np.float64)  # decimate requires float64
            i = group[i_col].values.astype(np.float64)
            if downsample_factor > 1 and len(u) > downsample_factor * 10:
                u = decimate(u, downsample_factor, ftype='fir', zero_phase=True)
                i = decimate(i, downsample_factor, ftype='fir', zero_phase=True)
            U_list.append(u.astype(np.float32))
            I_list.append(i.astype(np.float32))
            episode_ids.append(f"{file_tag}_event{event_no}")

    lengths = [len(u) for u in U_list]
    print(f"Loaded {len(U_list)} episodes (events) from {len(paths)} files.")
    return U_list, I_list, episode_ids, lengths


class EpisodeDataset(Dataset):
    """Each item is one full episode at its original length."""
    def __init__(self, I_list, U_list):
        self.I = [torch.tensor(i, dtype=torch.float32).unsqueeze(-1) for i in I_list]  # (T_i, 1)
        self.U = [torch.tensor(u, dtype=torch.float32).unsqueeze(-1) for u in U_list]

    def __len__(self):
        return len(self.I)

    def __getitem__(self, idx):
        return self.I[idx], self.U[idx]


def collate_variable_length(batch):
    """Pads sequences within the batch to the batch's max length and provides a binary mask."""
    i_seqs, u_seqs = zip(*batch)
    lengths = torch.tensor([len(s) for s in i_seqs], dtype=torch.long)

    i_padded = pad_sequence(i_seqs, batch_first=True, padding_value=0.0)  # (batch, T_max, 1)
    u_padded = pad_sequence(u_seqs, batch_first=True, padding_value=0.0)

    T_max = i_padded.size(1)
    mask = (torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)

    return i_padded, u_padded, lengths, mask


# ==========================================
# 2. Model: BiLSTM with variable-length handling
# ==========================================
class BiLSTMSurrogate(nn.Module):
    """Bidirectional LSTM Surrogate model for V-I mapping."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, i_seq, lengths):
        # i_seq: (batch, T_max, 1); lengths: (batch,)
        packed = pack_padded_sequence(i_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=i_seq.size(1))
        u_pred = self.fc(out)  # (batch, T_max, 1)
        return u_pred


def masked_mse(pred, target, mask):
    """MSE calculated only across valid (unpadded) timesteps."""
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1.0)


# ==========================================
# 3. Training & Evaluation
# ==========================================
def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training {model.__class__.__name__} (epochs={epochs}, device={device}) ---", flush=True)
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_bar:
        # ---- Train ----
        model.train()
        running, running_count = 0.0, 0
        t0 = time.time()
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{epochs} [train]",
            unit="batch",
            leave=False,
            position=1,
        )
        for bi, bu, lengths, mask in batch_bar:
            bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(bi, lengths)
            loss = masked_mse(pred, bu, mask)
            loss.backward()
            optimizer.step()
            running += loss.item() * bi.size(0)
            running_count += bi.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.5f}", seq_len=bi.size(1))
        train_loss = running / running_count
        batch_bar.close()

        # ---- Validate ----
        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1}/{epochs} [val]  ",
                unit="batch",
                leave=False,
                position=1,
            )
            for bi, bu, lengths, mask in val_bar:
                bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
                pred = model(bi, lengths)
                loss = masked_mse(pred, bu, mask)
                val_running += loss.item() * bi.size(0)
                val_count += bi.size(0)
                val_bar.set_postfix(val_loss=f"{loss.item():.5f}")
            val_bar.close()
        val_loss = val_running / val_count

        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            sec=f"{elapsed:.1f}s",
        )
        tqdm.write(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"Train MSE: {train_loss:.5f}  Val MSE: {val_loss:.5f}  "
            f"({elapsed:.1f}s)",
            file=sys.stdout,
        )

    epoch_bar.close()
    return model


def predict_per_episode(model, loader, device):
    """Returns predictions per episode, trimmed to true length."""
    model.eval()
    preds = []
    with torch.no_grad():
        for bi, _, lengths, _ in tqdm(loader, desc="Predicting", unit="batch"):
            bi = bi.to(device)
            out = model(bi, lengths).cpu().numpy()
            for b in range(out.shape[0]):
                preds.append(out[b, :lengths[b], :])
    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM Surrogate Model for V-I Curve Modeling")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing CSV files (default: auto-detect B6031600_event_all or data/)")
    parser.add_argument("--file_pattern", type=str, default="B6031600_*.csv",
                        help="Glob pattern for event CSV files (default: 'B6031600_*.csv')")
    parser.add_argument("--u_col", type=str, default="U1",
                        help="Target voltage column name (default: 'U1')")
    parser.add_argument("--i_col", type=str, default="I1",
                        help="Input current column name (default: 'I1')")
    parser.add_argument("--event_col", type=str, default="EventNo",
                        help="Event identifier column name (default: 'EventNo')")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsampling factor using anti-aliasing decimate (default: 1)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer (default: 0.001)")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Hidden units in BiLSTM (default: 32)")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of stacked BiLSTM layers (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate between layers (default: 0.2)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save output plots and metrics (default: 'results')")
    parser.add_argument("--zoom_window", type=int, default=150,
                        help="Number of timesteps to display in high-precision zoomed-in detail plot (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and load existing vi_bilstm_model.pt for immediate evaluation and plotting")
    return parser.parse_args()


def resolve_data_dir(user_specified_dir):
    if user_specified_dir:
        return user_specified_dir
    # Auto-detection priority:
    candidates = ["./B6031600_event_all", "./data", "."]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "B6031600_*.csv"))) > 0:
            return c
    return "./data"


def find_dynamic_zoom_window(signal, window_size):
    """Finds the window with the most rapid dynamic transitions (highest gradient magnitude)."""
    n = len(signal)
    if n <= window_size:
        return 0, n
    grad = np.abs(np.gradient(signal))
    kernel = np.ones(window_size)
    moving_activity = np.convolve(grad, kernel, mode="valid")
    start_idx = int(np.argmax(moving_activity))
    end_idx = start_idx + window_size
    return start_idx, end_idx


def plot_high_precision_comparison(
    u_true, u_pred, u_baseline, episode_id, length, label, metrics, output_dir, zoom_window=150
):
    """Generates a high-precision multi-panel comparison figure:
    1. Full Episode Overview with Highlighted Transient Zone & Performance Metrics Box
    2. High-Precision Zoomed-In Comparison with sample-level tracking and markers
    3. Instantaneous Absolute Error (Residuals) comparison
    """
    n_samples = len(u_true)
    time_idx = np.arange(n_samples)
    z_start, z_end = find_dynamic_zoom_window(u_true, min(zoom_window, n_samples))

    fig, (ax_main, ax_zoom, ax_err) = plt.subplots(
        3, 1, figsize=(14, 10), dpi=300,
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.2]}
    )

    # 1. Full Episode Overview
    ax_main.plot(time_idx, u_true, color="#1f1f1f", linewidth=1.8, alpha=0.9)
    ax_main.plot(time_idx, u_pred, color="#1f77b4", linestyle="--", linewidth=1.8)
    ax_main.plot(time_idx, u_baseline, color="#ff7f0e", linestyle=":", linewidth=1.4, alpha=0.8)

    # Highlight zoom region
    ax_main.axvspan(z_start, z_end, color="#d62728", alpha=0.15)
    ax_main.axvline(z_start, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_main.axvline(z_end, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)

    ax_main.set_title(f"Full Episode Overview ({label.capitalize()} Case)", fontsize=13, fontweight="bold")
    ax_main.set_ylabel("Voltage (U1)", fontsize=11)
    ax_main.grid(True, linestyle="--", alpha=0.5)

    # 2. High-Precision Zoomed-In Detail View
    zoom_t = time_idx[z_start:z_end]
    ax_zoom.plot(
        zoom_t, u_true[z_start:z_end],
        color="#1f1f1f", linewidth=2.0, marker="o", markersize=3.5, alpha=0.9
    )
    ax_zoom.plot(
        zoom_t, u_pred[z_start:z_end],
        color="#1f77b4", linestyle="--", linewidth=2.0, marker="x", markersize=4.0
    )
    ax_zoom.plot(
        zoom_t, u_baseline[z_start:z_end],
        color="#ff7f0e", linestyle=":", linewidth=1.6, marker=".", markersize=2.5, alpha=0.75
    )

    ax_zoom.set_title("Zoom in detail", fontsize=12, fontweight="bold")
    ax_zoom.set_xlim(z_start, z_end)
    ax_zoom.set_ylabel("Voltage (U1)", fontsize=11)
    ax_zoom.grid(True, linestyle="--", alpha=0.6)

    # 3. Residual / Absolute Error Comparison
    err_bilstm = np.abs(u_pred - u_true)
    err_baseline = np.abs(u_baseline - u_true)

    ax_err.plot(time_idx, err_baseline, color="#ff7f0e", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_err.plot(time_idx, err_bilstm, color="#2ca02c", linewidth=1.5, alpha=0.85)
    ax_err.fill_between(time_idx, 0, err_bilstm, color="#2ca02c", alpha=0.15)
    ax_err.axvspan(z_start, z_end, color="#d62728", alpha=0.10)

    ax_err.set_title("Instantaneous Absolute Error", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Sample Index", fontsize=11)
    ax_err.set_ylabel("|Error|", fontsize=11)
    ax_err.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"vi_bilstm_example_{label}_high_precision.png")
    plt.savefig(out_path, dpi=300)
    print(f"High-precision zoomed comparison plot saved to: {out_path}")
    plt.close()


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("Loading and splitting files into events...", flush=True)

    U_list, I_list, episode_ids, lengths = load_episode_files(
        data_dir, args.file_pattern, args.u_col, args.i_col, args.event_col, args.downsample
    )
    n_episodes = len(U_list)

    # Train / Val / Test split (file/episode level)
    idx = np.arange(n_episodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test episodes")

    # Fit scalers on TRAINING episodes only
    print("Fitting scalers...", flush=True)
    train_U_concat = np.concatenate([U_list[i] for i in train_idx]).reshape(-1, 1)
    train_I_concat = np.concatenate([I_list[i] for i in train_idx]).reshape(-1, 1)

    scaler_U = StandardScaler().fit(train_U_concat)
    scaler_I = StandardScaler().fit(train_I_concat)
    print("Scalers fitted. Scaling all episodes...", flush=True)

    def scale_list(arr_list, scaler):
        return [scaler.transform(a.reshape(-1, 1)).flatten() for a in arr_list]

    U_scaled_all = scale_list(U_list, scaler_U)
    I_scaled_all = scale_list(I_list, scaler_I)

    def subset(lst, indices):
        return [lst[i] for i in indices]

    train_ds = EpisodeDataset(subset(I_scaled_all, train_idx), subset(U_scaled_all, train_idx))
    val_ds = EpisodeDataset(subset(I_scaled_all, val_idx), subset(U_scaled_all, val_idx))
    test_ds = EpisodeDataset(subset(I_scaled_all, test_idx), subset(U_scaled_all, test_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Baseline: pointwise static linear regression (memory-less)
    lin_reg = LinearRegression()
    lin_reg.fit(train_I_concat, train_U_concat)

    # Initialize and train BiLSTM model
    model = BiLSTMSurrogate(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    model_path = os.path.join(args.output_dir, "vi_bilstm_model.pt")

    if args.eval_only:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"--eval_only was set, but model checkpoint '{model_path}' not found. "
                "Please run training without --eval_only first."
            )
        print(f"Loading existing trained model weights from: {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = train_model(model, train_loader, val_loader, device, args.epochs, lr=args.lr)
        # Save model weights
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to: {model_path}")

    # Evaluate on test set
    preds_scaled = predict_per_episode(model, test_loader, device)

    per_file_metrics_model = []
    per_file_metrics_baseline = []
    results_rows = []

    for local_i, global_i in tqdm(enumerate(test_idx), total=len(test_idx), desc="Evaluating", unit="ep"):
        u_true_scaled = U_scaled_all[global_i].reshape(-1, 1)
        i_true_scaled = I_scaled_all[global_i].reshape(-1, 1)

        u_true = scaler_U.inverse_transform(u_true_scaled).flatten()
        u_pred = scaler_U.inverse_transform(preds_scaled[local_i]).flatten()
        u_baseline = lin_reg.predict(scaler_I.inverse_transform(i_true_scaled)).flatten()

        # Model metrics
        model_mse = float(np.mean((u_pred - u_true) ** 2))
        model_rmse = float(np.sqrt(model_mse))
        model_mae = float(np.mean(np.abs(u_pred - u_true)))

        # Baseline metrics
        base_mse = float(np.mean((u_baseline - u_true) ** 2))
        base_rmse = float(np.sqrt(base_mse))
        base_mae = float(np.mean(np.abs(u_baseline - u_true)))

        # Variance and R2
        u_var = float(np.var(u_true))
        model_r2 = float(1.0 - (model_mse / u_var)) if u_var > 1e-12 else (1.0 if model_mse < 1e-12 else 0.0)
        base_r2 = float(1.0 - (base_mse / u_var)) if u_var > 1e-12 else (1.0 if base_mse < 1e-12 else 0.0)

        # Fit Accuracy (NRMSE based): max(0, 1 - RMSE / span) * 100%
        u_span = float(np.ptp(u_true))
        if u_span > 1e-12:
            model_acc = float(max(0.0, 1.0 - model_rmse / u_span) * 100.0)
            base_acc = float(max(0.0, 1.0 - base_rmse / u_span) * 100.0)
            tol_val = 0.02 * u_span
            model_tol_acc = float(np.mean(np.abs(u_pred - u_true) <= tol_val) * 100.0)
            base_tol_acc = float(np.mean(np.abs(u_baseline - u_true) <= tol_val) * 100.0)
        else:
            model_acc = 100.0 if model_rmse < 1e-12 else 0.0
            base_acc = 100.0 if base_rmse < 1e-12 else 0.0
            model_tol_acc = 100.0 if model_rmse < 1e-12 else 0.0
            base_tol_acc = 100.0 if base_rmse < 1e-12 else 0.0

        model_ep_metrics = {
            "bilstm_mse": model_mse,
            "bilstm_rmse": model_rmse,
            "bilstm_mae": model_mae,
            "bilstm_r2": model_r2,
            "bilstm_acc": model_acc,
            "bilstm_tol_acc": model_tol_acc,
        }
        base_ep_metrics = {
            "baseline_mse": base_mse,
            "baseline_rmse": base_rmse,
            "baseline_mae": base_mae,
            "baseline_r2": base_r2,
            "baseline_acc": base_acc,
            "baseline_tol_acc": base_tol_acc,
        }

        per_file_metrics_model.append(model_ep_metrics)
        per_file_metrics_baseline.append(base_ep_metrics)

        results_rows.append({
            "episode_id": episode_ids[global_i],
            "length": lengths[global_i],
            **model_ep_metrics,
            **base_ep_metrics,
        })

    model_df = pd.DataFrame(per_file_metrics_model)
    base_df = pd.DataFrame(per_file_metrics_baseline)

    mean_m_mse, mean_b_mse = float(model_df["bilstm_mse"].mean()), float(base_df["baseline_mse"].mean())
    mean_m_rmse, mean_b_rmse = float(model_df["bilstm_rmse"].mean()), float(base_df["baseline_rmse"].mean())
    mean_m_mae, mean_b_mae = float(model_df["bilstm_mae"].mean()), float(base_df["baseline_mae"].mean())
    mean_m_r2, mean_b_r2 = float(model_df["bilstm_r2"].mean()), float(base_df["baseline_r2"].mean())
    mean_m_acc, mean_b_acc = float(model_df["bilstm_acc"].mean()), float(base_df["baseline_acc"].mean())
    mean_m_tol, mean_b_tol = float(model_df["bilstm_tol_acc"].mean()), float(base_df["baseline_tol_acc"].mean())

    mse_imp = (1.0 - mean_m_mse / mean_b_mse) * 100.0 if mean_b_mse > 0 else float("nan")
    rmse_imp = (1.0 - mean_m_rmse / mean_b_rmse) * 100.0 if mean_b_rmse > 0 else float("nan")
    mae_imp = (1.0 - mean_m_mae / mean_b_mae) * 100.0 if mean_b_mae > 0 else float("nan")

    print("\n" + "=" * 82)
    print("        Test-Set Performance & Accuracy Summary (BiLSTM vs. Baseline)")
    print("=" * 82)
    print(f"{'Evaluation Metric':<26} | {'BiLSTM Model':<16} | {'Linear Baseline':<16} | {'Comparison':<16}")
    print("-" * 82)
    print(f"{'Mean Fit Accuracy (%)':<26} | {mean_m_acc:<16.2f}% | {mean_b_acc:<16.2f}% | {mean_m_acc - mean_b_acc:+.2f}%")
    print(f"{'2% Tolerance Acc (%)':<26} | {mean_m_tol:<16.2f}% | {mean_b_tol:<16.2f}% | {mean_m_tol - mean_b_tol:+.2f}%")
    print(f"{'Mean R² Score':<26} | {mean_m_r2:<16.4f} | {mean_b_r2:<16.4f} | {mean_m_r2 - mean_b_r2:+.4f}")
    print(f"{'Mean RMSE':<26} | {mean_m_rmse:<16.5f} | {mean_b_rmse:<16.5f} | {rmse_imp:+.1f}% error red.")
    print(f"{'Mean MAE':<26} | {mean_m_mae:<16.5f} | {mean_b_mae:<16.5f} | {mae_imp:+.1f}% error red.")
    print(f"{'Mean MSE':<26} | {mean_m_mse:<16.5f} | {mean_b_mse:<16.5f} | {mse_imp:+.1f}% error red.")
    print(f"{'Worst Episode MSE':<26} | {float(model_df['bilstm_mse'].max()):<16.5f} | {float(base_df['baseline_mse'].max()):<16.5f} | -")
    print("=" * 82)

    results_csv_path = os.path.join(args.output_dir, "vi_bilstm_test_results.csv")
    results_df = pd.DataFrame(results_rows)
    try:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Per-episode test results and accuracy saved to: {results_csv_path}")
    except PermissionError:
        print(f"[Warning] Could not write to '{results_csv_path}'. Is it open in Excel? Skipping CSV save and continuing to plot...", flush=True)

    # Plot sample test episodes
    per_file_mse = model_df["bilstm_mse"].values
    example_local = {
        "typical": int(np.argsort(per_file_mse)[len(per_file_mse) // 2]),
        "worst_case": int(np.argmax(per_file_mse)),
    }

    for label, local_idx in example_local.items():
        global_i = test_idx[local_idx]
        u_true = scaler_U.inverse_transform(U_scaled_all[global_i].reshape(-1, 1)).flatten()
        u_pred = scaler_U.inverse_transform(preds_scaled[local_idx]).flatten()
        i_vals = scaler_I.inverse_transform(I_scaled_all[global_i].reshape(-1, 1)).flatten()
        u_baseline = lin_reg.predict(i_vals.reshape(-1, 1)).flatten()

        eid = episode_ids[global_i]
        ep_len = lengths[global_i]
        ep_metrics = {**per_file_metrics_model[local_idx], **per_file_metrics_baseline[local_idx]}

        # 1. Standard overview plot
        plt.figure(figsize=(14, 5))
        plt.plot(u_true, color="black", linewidth=2)
        plt.plot(u_pred, color="tab:blue", linestyle="--", linewidth=2)
        plt.plot(u_baseline, color="gray", linestyle=":", linewidth=1.5)
        plt.title(f"V-I Curve Comparison ({label.capitalize()} Case)")
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (U1)")
        plt.grid(True)
        plt.tight_layout()
        out_name = os.path.join(args.output_dir, f"vi_bilstm_example_{label}.png")
        plt.savefig(out_name, dpi=300)
        print(f"Plot saved as: {out_name}")
        plt.close()

        # 2. High-precision zoomed-in comparison plot
        plot_high_precision_comparison(
            u_true=u_true,
            u_pred=u_pred,
            u_baseline=u_baseline,
            episode_id=eid,
            length=ep_len,
            label=label,
            metrics=ep_metrics,
            output_dir=args.output_dir,
            zoom_window=args.zoom_window,
        )


if __name__ == "__main__":
    main()