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


def summarize_and_print_metrics(model_df, base_df):
    """
    統計所有測試事件的平均指標，並在終端打印格式化對比報表（BiLSTM vs. Linear Baseline）。
    """
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
