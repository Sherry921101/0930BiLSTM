"""
BiLSTM.py: 電壓-電流 (V-I) 特性曲線動態代理模型的主執行入口。
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    load_episode_files,
    EpisodeDataset,
    collate_variable_length,
    resolve_data_dir,
)
from src.model import BiLSTMSurrogate, RNNSurrogate, LSTMSurrogate, GRUSurrogate
from src.trainer import (
    masked_mse,
    train_model,
    predict_per_episode,
)
from src.metrics import (
    compute_episode_metrics,
    summarize_and_print_metrics,
)
from src.visualization import (
    find_dynamic_zoom_window,
    plot_overview,
    plot_high_precision_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Surrogate Models for V-I Curve Modeling (Single Dataset)")
    parser.add_argument("--data_dir", type=str, default="26011200_event_all", help="Directory for data")
    parser.add_argument("--file_pattern", type=str, default="26011200*.csv", help="Pattern for files")
    parser.add_argument("--u_col", type=str, default="U12", help="Voltage column")
    parser.add_argument("--i_col", type=str, default="I1", help="Current column")
    parser.add_argument("--event_col", type=str, default="EventNo")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--zoom_window", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Data from {data_dir}...")
    U_list, I_list, episode_ids, lengths = load_episode_files(
        data_dir, args.file_pattern, args.u_col, args.i_col, args.event_col, args.downsample
    )
    n_episodes = len(U_list)

    # 80% Train, 10% Val, 10% Test
    idx = np.arange(n_episodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test episodes")

    train_U_concat = np.concatenate([U_list[i] for i in train_idx]).reshape(-1, 1)
    train_I_concat = np.concatenate([I_list[i] for i in train_idx]).reshape(-1, 1)

    scaler_U = StandardScaler().fit(train_U_concat)
    scaler_I = StandardScaler().fit(train_I_concat)

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

    lin_reg = LinearRegression()
    lin_reg.fit(train_I_concat, train_U_concat)

    model_classes = {
        "RNN": RNNSurrogate,
        "LSTM": LSTMSurrogate,
        "GRU": GRUSurrogate,
        "BiLSTM": BiLSTMSurrogate
    }

    models = {}
    preds_scaled_all = {}

    for m_name, m_cls in model_classes.items():
        print(f"\n--- Training {m_name} ---")
        m = m_cls(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
        model_path = os.path.join(args.output_dir, f"vi_{m_name.lower()}_model.pt")

        if args.eval_only:
            m.load_state_dict(torch.load(model_path, map_location=device))
        else:
            m = train_model(m, train_loader, val_loader, device, args.epochs, lr=args.lr)
            torch.save(m.state_dict(), model_path)
            print(f"Trained {m_name} model saved to: {model_path}")
        
        models[m_name] = m
        preds_scaled_all[m_name] = predict_per_episode(m, test_loader, device)

    per_file_metrics = {m: [] for m in model_classes.keys()}
    per_file_metrics["Baseline"] = []
    results_rows = []

    for local_i, global_i in tqdm(enumerate(test_idx), total=len(test_idx), desc="Evaluating"):
        u_true_scaled = U_scaled_all[global_i].reshape(-1, 1)
        i_true_scaled = I_scaled_all[global_i].reshape(-1, 1)

        u_true = scaler_U.inverse_transform(u_true_scaled).flatten()
        u_baseline = lin_reg.predict(scaler_I.inverse_transform(i_true_scaled)).flatten()
        
        row = {"episode_id": episode_ids[global_i], "length": lengths[global_i]}
        base_ep_metrics = compute_episode_metrics(u_true, u_baseline, prefix="baseline")
        per_file_metrics["Baseline"].append(base_ep_metrics)
        row.update(base_ep_metrics)
        
        for m_name in model_classes.keys():
            u_pred = scaler_U.inverse_transform(preds_scaled_all[m_name][local_i]).flatten()
            m_metrics = compute_episode_metrics(u_true, u_pred, prefix=m_name.lower())
            per_file_metrics[m_name].append(m_metrics)
            row.update(m_metrics)
            
        results_rows.append(row)

    # Print summary
    all_metrics_dfs = {m_name: pd.DataFrame(metrics_list) for m_name, metrics_list in per_file_metrics.items()}
    summarize_and_print_metrics(all_metrics_dfs)

    results_csv_path = os.path.join(args.output_dir, "vi_models_test_results.csv")
    results_df = pd.DataFrame(results_rows)
    try:
        results_df.to_csv(results_csv_path, index=False)
    except PermissionError:
        pass

    per_file_mse = pd.DataFrame(per_file_metrics["BiLSTM"])["bilstm_mse"].values
    example_local = {
        "typical": int(np.argsort(per_file_mse)[len(per_file_mse) // 2]),
        "worst_case": int(np.argmax(per_file_mse)),
    }

    for label, local_idx in example_local.items():
        global_i = test_idx[local_idx]
        u_true = scaler_U.inverse_transform(U_scaled_all[global_i].reshape(-1, 1)).flatten()
        i_vals = scaler_I.inverse_transform(I_scaled_all[global_i].reshape(-1, 1)).flatten()
        
        predictions = {"Baseline": lin_reg.predict(i_vals.reshape(-1, 1)).flatten()}
        for m_name in model_classes.keys():
            predictions[m_name] = scaler_U.inverse_transform(preds_scaled_all[m_name][local_idx]).flatten()
            
        eid = episode_ids[global_i]
        ep_len = lengths[global_i]

        plot_overview(u_true, predictions, label, args.output_dir)
        plot_high_precision_comparison(
            u_true=u_true,
            predictions=predictions,
            episode_id=eid,
            length=ep_len,
            label=label,
            metrics={},
            output_dir=args.output_dir,
            zoom_window=args.zoom_window,
        )

if __name__ == "__main__":
    main()