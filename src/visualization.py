import os
import matplotlib.pyplot as plt
import numpy as np


def find_dynamic_zoom_window(signal, window_size):
    """
    自動特徵檢測演算法：
    透過計算訊號梯度的絕對值並進行滑動窗口卷積（Moving Sum），
    自動定位出整條波形中動態變化最劇烈、跳變最快速的暫態時間區段。
    """
    n = len(signal)
    if n <= window_size:
        return 0, n
    grad = np.abs(np.gradient(signal))
    kernel = np.ones(window_size)
    moving_activity = np.convolve(grad, kernel, mode="valid")
    start_idx = int(np.argmax(moving_activity))
    end_idx = start_idx + window_size
    return start_idx, end_idx


def plot_overview(u_true, predictions, label, output_dir):
    """輸出標準全時域電壓曲線對比總覽圖。"""
    plt.figure(figsize=(14, 5))
    plt.plot(u_true, color="black", linewidth=2, label="True")
    
    colors = {"Baseline": "gray", "RNN": "tab:purple", "LSTM": "tab:cyan", "GRU": "tab:brown", "BiLSTM": "tab:blue"}
    styles = {"Baseline": ":", "RNN": "-.", "LSTM": "-.", "GRU": "-.", "BiLSTM": "--"}
    
    for model_name, u_pred in predictions.items():
        c = colors.get(model_name, "tab:red")
        s = styles.get(model_name, "-")
        plt.plot(u_pred, color=c, linestyle=s, linewidth=1.5, label=model_name, alpha=0.8)
        
    plt.title(f"V-I Curve Comparison ({label.capitalize()} Case)")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (U1)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_name = os.path.join(output_dir, f"vi_models_example_{label}.png")
    plt.savefig(out_name, dpi=300)
    print(f"Overview plot saved to: {out_name}")
    plt.close()


def plot_high_precision_comparison(
    u_true, predictions, episode_id, length, label, metrics, output_dir, zoom_window=150
):
    """
    生成高精度評估比較圖（300 DPI）
    """
    n_samples = len(u_true)
    time_idx = np.arange(n_samples)
    z_start, z_end = find_dynamic_zoom_window(u_true, min(zoom_window, n_samples))

    fig, (ax_main, ax_zoom, ax_err) = plt.subplots(
        3, 1, figsize=(14, 10), dpi=300,
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.2]}
    )

    colors = {"Baseline": "#ff7f0e", "RNN": "#9467bd", "LSTM": "#17becf", "GRU": "#8c564b", "BiLSTM": "#1f77b4"}
    styles = {"Baseline": ":", "RNN": "-.", "LSTM": "-.", "GRU": "-.", "BiLSTM": "--"}

    # 1. Full Episode Overview
    ax_main.plot(time_idx, u_true, color="#1f1f1f", linewidth=1.8, alpha=0.9, label="True")
    for model_name, u_pred in predictions.items():
        c = colors.get(model_name, "#2ca02c")
        s = styles.get(model_name, "-")
        lw = 1.8 if model_name == "BiLSTM" else 1.4
        ax_main.plot(time_idx, u_pred, color=c, linestyle=s, linewidth=lw, label=model_name, alpha=0.8)

    ax_main.axvspan(z_start, z_end, color="#d62728", alpha=0.15)
    ax_main.axvline(z_start, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_main.axvline(z_end, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)

    ax_main.set_title(f"Full Episode Overview ({label.capitalize()} Case)", fontsize=13, fontweight="bold")
    ax_main.set_ylabel("Voltage (U1)", fontsize=11)
    ax_main.grid(True, linestyle="--", alpha=0.5)
    ax_main.legend(loc="upper right")

    # 2. High-Precision Zoomed-In Detail View
    zoom_t = time_idx[z_start:z_end]
    ax_zoom.plot(
        zoom_t, u_true[z_start:z_end],
        color="#1f1f1f", linewidth=2.0, marker="o", markersize=3.5, alpha=0.9, label="True"
    )
    for model_name, u_pred in predictions.items():
        c = colors.get(model_name, "#2ca02c")
        s = styles.get(model_name, "-")
        lw = 2.0 if model_name == "BiLSTM" else 1.6
        marker = "x" if model_name == "BiLSTM" else "."
        ax_zoom.plot(
            zoom_t, u_pred[z_start:z_end],
            color=c, linestyle=s, linewidth=lw, marker=marker, markersize=3.0, alpha=0.8, label=model_name
        )

    ax_zoom.set_title("Zoomed-In Detail (Dynamic Transient Region)", fontsize=12, fontweight="bold")
    ax_zoom.set_xlim(z_start, z_end)
    ax_zoom.set_ylabel("Voltage (U1)", fontsize=11)
    ax_zoom.grid(True, linestyle="--", alpha=0.6)
    ax_zoom.legend(loc="upper right")

    # 3. Residual / Absolute Error Comparison
    for model_name, u_pred in predictions.items():
        c = colors.get(model_name, "#2ca02c")
        s = styles.get(model_name, "-")
        err = np.abs(u_pred - u_true)
        if model_name == "BiLSTM":
            ax_err.plot(time_idx, err, color=c, linewidth=1.5, alpha=0.85, label=f"{model_name} Error")
            ax_err.fill_between(time_idx, 0, err, color=c, alpha=0.15)
        else:
            ax_err.plot(time_idx, err, color=c, linestyle=s, linewidth=1.2, alpha=0.7, label=f"{model_name} Error")
            
    ax_err.axvspan(z_start, z_end, color="#d62728", alpha=0.10)

    ax_err.set_title("Instantaneous Absolute Error", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Sample Index", fontsize=11)
    ax_err.set_ylabel("|Error|", fontsize=11)
    ax_err.grid(True, linestyle="--", alpha=0.5)
    ax_err.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"vi_models_example_{label}_high_precision.png")
    plt.savefig(out_path, dpi=300)
    print(f"High-precision zoomed comparison plot saved to: {out_path}")
    plt.close()
