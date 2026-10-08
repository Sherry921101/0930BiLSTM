"""
transfer.py: 預訓練與微調的遷移學習腳本
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
)
from src.model import BiLSTMSurrogate
from src.trainer import (
    train_model,
    predict_per_episode,
)
from src.metrics import (
    compute_episode_metrics,
    summarize_and_print_metrics,
)
from src.visualization import (
    plot_overview,
    plot_high_precision_comparison,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer Learning for BiLSTM")
    parser.add_argument("--pretrain_dir", type=str, default="B6031600_event_all", help="Pretrain data dir")
    parser.add_argument("--pretrain_pattern", type=str, default="*.csv", help="Pretrain file pattern")
    parser.add_argument("--finetune_dir", type=str, default="26011200_event_all", help="Finetune data dir")
    parser.add_argument("--finetune_pattern", type=str, default="*.csv", help="Finetune file pattern")
    
    parser.add_argument("--pretrain_u_col", type=str, default="U1")
    parser.add_argument("--finetune_u_col", type=str, default="U12")
    parser.add_argument("--i_col", type=str, default="I1")
    parser.add_argument("--event_col", type=str, default="EventNo")
    parser.add_argument("--downsample", type=int, default=1)
    
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--output_dir", type=str, default="results_transfer")
    parser.add_argument("--zoom_window", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def scale_list(arr_list, scaler):
    return [scaler.transform(a.reshape(-1, 1)).flatten() for a in arr_list]

def subset(lst, indices):
    return [lst[i] for i in indices]

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. 預訓練階段 (Pretraining)
    # ==========================================
    print(f"Loading pretrain data from {args.pretrain_dir}...")
    pt_U, pt_I, pt_ep_ids, pt_lens = load_episode_files(
        args.pretrain_dir, args.pretrain_pattern, args.pretrain_u_col, args.i_col, args.event_col, args.downsample
    )
    
    pt_n_episodes = len(pt_U)
    pt_idx = np.arange(pt_n_episodes)
    # Pretrain 80% train, 20% val
    pt_train_idx, pt_val_idx = train_test_split(pt_idx, test_size=0.2, random_state=args.seed)
    
    print(f"Pretrain split: {len(pt_train_idx)} train / {len(pt_val_idx)} val episodes")
    
    pt_train_U_concat = np.concatenate([pt_U[i] for i in pt_train_idx]).reshape(-1, 1)
    pt_train_I_concat = np.concatenate([pt_I[i] for i in pt_train_idx]).reshape(-1, 1)
    
    pt_scaler_U = StandardScaler().fit(pt_train_U_concat)
    pt_scaler_I = StandardScaler().fit(pt_train_I_concat)
    
    pt_U_scaled = scale_list(pt_U, pt_scaler_U)
    pt_I_scaled = scale_list(pt_I, pt_scaler_I)
    
    pt_train_ds = EpisodeDataset(subset(pt_I_scaled, pt_train_idx), subset(pt_U_scaled, pt_train_idx))
    pt_val_ds = EpisodeDataset(subset(pt_I_scaled, pt_val_idx), subset(pt_U_scaled, pt_val_idx))
    
    pt_train_loader = DataLoader(pt_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_length)
    pt_val_loader = DataLoader(pt_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    
    model = BiLSTMSurrogate(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    pretrain_model_path = os.path.join(args.output_dir, "pretrained_model.pt")
    
    print("Starting Pretraining...")
    model = train_model(model, pt_train_loader, pt_val_loader, device, args.pretrain_epochs, lr=args.lr)
    torch.save(model.state_dict(), pretrain_model_path)
    print(f"Pretrained model saved to: {pretrain_model_path}")
    
    # ==========================================
    # 2. 微調階段 (Fine-tuning)
    # ==========================================
    print(f"Loading finetune data from {args.finetune_dir}...")
    ft_U, ft_I, ft_ep_ids, ft_lens = load_episode_files(
        args.finetune_dir, args.finetune_pattern, args.finetune_u_col, args.i_col, args.event_col, args.downsample
    )
    
    ft_n_episodes = len(ft_U)
    ft_idx = np.arange(ft_n_episodes)
    
    # Finetune 80% train, 10% val, 10% test
    ft_train_idx, ft_temp_idx = train_test_split(ft_idx, test_size=0.2, random_state=args.seed)
    ft_val_idx, ft_test_idx = train_test_split(ft_temp_idx, test_size=0.5, random_state=args.seed)
    
    print(f"Finetune split: {len(ft_train_idx)} train / {len(ft_val_idx)} val / {len(ft_test_idx)} test episodes")
    
    ft_train_U_concat = np.concatenate([ft_U[i] for i in ft_train_idx]).reshape(-1, 1)
    ft_train_I_concat = np.concatenate([ft_I[i] for i in ft_train_idx]).reshape(-1, 1)
    
    ft_scaler_U = StandardScaler().fit(ft_train_U_concat)
    ft_scaler_I = StandardScaler().fit(ft_train_I_concat)
    
    ft_U_scaled = scale_list(ft_U, ft_scaler_U)
    ft_I_scaled = scale_list(ft_I, ft_scaler_I)
    
    ft_train_ds = EpisodeDataset(subset(ft_I_scaled, ft_train_idx), subset(ft_U_scaled, ft_train_idx))
    ft_val_ds = EpisodeDataset(subset(ft_I_scaled, ft_val_idx), subset(ft_U_scaled, ft_val_idx))
    ft_test_ds = EpisodeDataset(subset(ft_I_scaled, ft_test_idx), subset(ft_U_scaled, ft_test_idx))
    
    ft_train_loader = DataLoader(ft_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_length)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    ft_test_loader = DataLoader(ft_test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    
    print("Starting Fine-tuning...")
    model = train_model(model, ft_train_loader, ft_val_loader, device, args.finetune_epochs, lr=args.finetune_lr)
    finetuned_model_path = os.path.join(args.output_dir, "finetuned_model.pt")
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Finetuned model saved to: {finetuned_model_path}")
    
    # ==========================================
    # 3. 評估與輸出 (Evaluation)
    # ==========================================
    lin_reg = LinearRegression()
    lin_reg.fit(ft_train_I_concat, ft_train_U_concat)
    
    print("Evaluating on test set...")
    preds_scaled = predict_per_episode(model, ft_test_loader, device)
    
    per_file_metrics_model = []
    per_file_metrics_baseline = []
    results_rows = []
    
    for local_i, global_i in tqdm(enumerate(ft_test_idx), total=len(ft_test_idx), desc="Evaluating"):
        u_true_scaled = ft_U_scaled[global_i].reshape(-1, 1)
        i_true_scaled = ft_I_scaled[global_i].reshape(-1, 1)
        
        u_true = ft_scaler_U.inverse_transform(u_true_scaled).flatten()
        u_pred = ft_scaler_U.inverse_transform(preds_scaled[local_i]).flatten()
        u_baseline = lin_reg.predict(ft_scaler_I.inverse_transform(i_true_scaled)).flatten()
        
        model_ep_metrics = compute_episode_metrics(u_true, u_pred, prefix="bilstm")
        base_ep_metrics = compute_episode_metrics(u_true, u_baseline, prefix="baseline")
        
        per_file_metrics_model.append(model_ep_metrics)
        per_file_metrics_baseline.append(base_ep_metrics)
        
        results_rows.append({
            "episode_id": ft_ep_ids[global_i],
            "length": ft_lens[global_i],
            **model_ep_metrics,
            **base_ep_metrics,
        })
        
    model_df = pd.DataFrame(per_file_metrics_model)
    base_df = pd.DataFrame(per_file_metrics_baseline)
    
    summarize_and_print_metrics(model_df, base_df)
    
    results_csv_path = os.path.join(args.output_dir, "transfer_test_results.csv")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Test results saved to: {results_csv_path}")
    
    if len(per_file_metrics_model) > 0:
        per_file_mse = model_df["bilstm_mse"].values
        example_local = {
            "typical": int(np.argsort(per_file_mse)[len(per_file_mse) // 2]),
            "worst_case": int(np.argmax(per_file_mse)),
        }
        
        for label, local_idx in example_local.items():
            global_i = ft_test_idx[local_idx]
            u_true = ft_scaler_U.inverse_transform(ft_U_scaled[global_i].reshape(-1, 1)).flatten()
            u_pred = ft_scaler_U.inverse_transform(preds_scaled[local_idx]).flatten()
            i_vals = ft_scaler_I.inverse_transform(ft_I_scaled[global_i].reshape(-1, 1)).flatten()
            u_baseline = lin_reg.predict(i_vals.reshape(-1, 1)).flatten()
            
            eid = ft_ep_ids[global_i]
            ep_len = ft_lens[global_i]
            ep_metrics = {**per_file_metrics_model[local_idx], **per_file_metrics_baseline[local_idx]}
            
            plot_overview(u_true, u_pred, u_baseline, label, args.output_dir)
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
