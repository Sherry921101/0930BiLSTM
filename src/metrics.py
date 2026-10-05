import numpy as np


def compute_episode_metrics(u_true, u_pred, prefix="bilstm"):
    """
    計算單一事件的完整工程評估指標：
    - MSE, RMSE, MAE
    - 決定係數 R^2
    - 擬合精度 Fit Accuracy (基於 NRMSE)
    - 容差精度 Tolerance Accuracy (落在全幅值 2% 誤差帶內的採樣點百分比)
    """
    mse = float(np.mean((u_pred - u_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(u_pred - u_true)))

    u_var = float(np.var(u_true))
    r2 = float(1.0 - (mse / u_var)) if u_var > 1e-12 else (1.0 if mse < 1e-12 else 0.0)

    u_span = float(np.ptp(u_true))
    if u_span > 1e-12:
        acc = float(max(0.0, 1.0 - rmse / u_span) * 100.0)
        tol_val = 0.02 * u_span
        tol_acc = float(np.mean(np.abs(u_pred - u_true) <= tol_val) * 100.0)
    else:
        acc = 100.0 if rmse < 1e-12 else 0.0
        tol_acc = 100.0 if rmse < 1e-12 else 0.0

    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
        f"{prefix}_acc": acc,
        f"{prefix}_tol_acc": tol_acc,
    }


def summarize_and_print_metrics(all_metrics_dfs):
    """
    統計所有測試事件的平均指標，並在終端打印多模型對比報表。
    all_metrics_dfs: dict, mapping model name (e.g., "Baseline", "RNN", ...) to its DataFrame.
    """
    # 預期 prefix 是轉為小寫的模型名稱
    models = list(all_metrics_dfs.keys())
    
    print("\n" + "=" * 90)
    print("                    Test-Set Performance & Accuracy Summary")
    print("=" * 90)
    
    # Header
    header = f"{'Evaluation Metric':<20} | " + " | ".join([f"{m:<10}" for m in models])
    print(header)
    print("-" * 90)
    
    metrics = [
        ("Mean Fit Acc (%)", "acc", "{:.2f}"),
        ("2% Tolerance Acc (%)", "tol_acc", "{:.2f}"),
        ("Mean R² Score", "r2", "{:.4f}"),
        ("Mean RMSE", "rmse", "{:.5f}"),
        ("Mean MAE", "mae", "{:.5f}"),
        ("Mean MSE", "mse", "{:.5f}"),
        ("Worst MSE", "mse", "{:.5f}"),
    ]
    
    for metric_name, suffix, fmt in metrics:
        row_str = f"{metric_name:<20} | "
        vals = []
        for m in models:
            df = all_metrics_dfs[m]
            prefix = m.lower()
            col_name = f"{prefix}_{suffix}"
            
            if metric_name == "Worst MSE":
                val = float(df[col_name].max())
            else:
                val = float(df[col_name].mean())
            vals.append(fmt.format(val))
        row_str += " | ".join([f"{v:<10}" for v in vals])
        print(row_str)
        
    print("=" * 90)
