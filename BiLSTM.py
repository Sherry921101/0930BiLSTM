"""
BiLSTM.py: 電壓-電流 (V-I) 特性曲線動態代理模型的主執行入口。
- 資料載入與批次預處理: src.data
- 模型架構:  src.model
- 訓練與推論執行引擎:  src.trainer
- 指標評估:  src.metrics
- 視覺化繪圖:  src.visualization
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

# 匯入模組化子套件 (src)
from src.data import (
    load_episode_files,
    EpisodeDataset,
    collate_variable_length,
    resolve_data_dir,
)
from src.model import BiLSTMSurrogate
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
    """解析命令列參數與模型超參數。"""
    parser = argparse.ArgumentParser(description="BiLSTM Surrogate Model for V-I Curve Modeling")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing CSV files (default: auto-detect B6031600_event_all or data/)")
    parser.add_argument("--file_pattern", type=str, default="B6031600*.csv",
                        help="Glob pattern for event CSV files (default: 'B6031600*.csv')")
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


def main():
    args = parse_args()

    # 1. 隨機種子設定 (確保結果可重現)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. 自動解析資料目錄與建立輸出路徑
    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # 3. 讀取並依據事件編號 (EventNo) 切分獨立事件
    print("Loading and splitting files into events...", flush=True)
    U_list, I_list, episode_ids, lengths = load_episode_files(
        data_dir, args.file_pattern, args.u_col, args.i_col, args.event_col, args.downsample
    )
    n_episodes = len(U_list)

    # 4. 以「事件編號（Episode 級別）」進行切分 (Train 80% / Val 10% / Test 10%)
    idx = np.arange(n_episodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test episodes")

    # 5. 標準化（僅利用訓練集擬合 StandardScaler，避免資料洩漏）
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

    # 6. 封裝 Dataset 與 DataLoader (支援變長批次填充與遮罩)
    train_ds = EpisodeDataset(subset(I_scaled_all, train_idx), subset(U_scaled_all, train_idx))
    val_ds = EpisodeDataset(subset(I_scaled_all, val_idx), subset(U_scaled_all, val_idx))
    test_ds = EpisodeDataset(subset(I_scaled_all, test_idx), subset(U_scaled_all, test_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 7. 擬合無記憶的靜態點對點線性回歸作為對照組 (Baseline)
    lin_reg = LinearRegression()
    lin_reg.fit(train_I_concat, train_U_concat)

    # 8. 初始化 BiLSTM 代理模型
    model = BiLSTMSurrogate(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    model_path = os.path.join(args.output_dir, "vi_bilstm_model.pt")

    # 訓練或載入既有權重
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
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to: {model_path}")

    # 9. 測試集批量推論
    preds_scaled = predict_per_episode(model, test_loader, device)

    per_file_metrics_model = []
    per_file_metrics_baseline = []
    results_rows = []

    # 10. 測試集指標逐事件計算
    for local_i, global_i in tqdm(enumerate(test_idx), total=len(test_idx), desc="Evaluating", unit="ep"):
        u_true_scaled = U_scaled_all[global_i].reshape(-1, 1)
        i_true_scaled = I_scaled_all[global_i].reshape(-1, 1)

        # 反標準化回真實物理量
        u_true = scaler_U.inverse_transform(u_true_scaled).flatten()
        u_pred = scaler_U.inverse_transform(preds_scaled[local_i]).flatten()
        u_baseline = lin_reg.predict(scaler_I.inverse_transform(i_true_scaled)).flatten()

        # 計算指標
        model_ep_metrics = compute_episode_metrics(u_true, u_pred, prefix="bilstm")
        base_ep_metrics = compute_episode_metrics(u_true, u_baseline, prefix="baseline")

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

    # 印製對比統計總表
    summarize_and_print_metrics(model_df, base_df)

    # 儲存逐事件 CSV 評估檔案
    results_csv_path = os.path.join(args.output_dir, "vi_bilstm_test_results.csv")
    results_df = pd.DataFrame(results_rows)
    try:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Per-episode test results and accuracy saved to: {results_csv_path}")
    except PermissionError:
        print(f"[Warning] Could not write to '{results_csv_path}'. Is it open in Excel? Skipping CSV save and continuing to plot...", flush=True)

    # 11. 自動挑選典型樣本 (Typical) 與最差案例 (Worst Case) 繪製成果圖
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

        # 1. 繪製全時域標準總覽圖
        plot_overview(u_true, u_pred, u_baseline, label, args.output_dir)

        # 2. 繪製高精度暫態與殘差三合一圖
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