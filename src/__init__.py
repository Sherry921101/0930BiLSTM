"""
V-I Curve Modeling (電壓-電流曲線動態代理模型) 核心模組套件
"""

from .data import (
    load_episode_files,
    EpisodeDataset,
    collate_variable_length,
    resolve_data_dir,
)
from .model import BiLSTMSurrogate
from .trainer import (
    masked_mse,
    train_model,
    predict_per_episode,
)
from .metrics import (
    compute_episode_metrics,
    summarize_and_print_metrics,
)
from .visualization import (
    find_dynamic_zoom_window,
    plot_overview,
    plot_high_precision_comparison,
)

__all__ = [
    "load_episode_files",
    "EpisodeDataset",
    "collate_variable_length",
    "resolve_data_dir",
    "BiLSTMSurrogate",
    "masked_mse",
    "train_model",
    "predict_per_episode",
    "compute_episode_metrics",
    "summarize_and_print_metrics",
    "find_dynamic_zoom_window",
    "plot_overview",
    "plot_high_precision_comparison",
]
