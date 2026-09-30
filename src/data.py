import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import decimate
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def resolve_data_dir(user_specified_dir):
    """
    自動偵測資料集目錄路徑：
    優先權：使用者指定路徑 -> './B6031600_event_all' -> './data' -> 當前目錄。
    """
    if user_specified_dir:
        return user_specified_dir
    candidates = ["./B6031600_event_all", "./data", "."]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "B6031600_*.csv"))) > 0:
            return c
    return "./data"


def load_episode_files(file_dir, pattern, u_col, i_col, event_col, downsample_factor=1):
    """
    讀取原始 CSV 檔案並依照事件編號（event_col）切分為獨立的事件序列（Episodes）。
    
    參數:
    - file_dir: 存放 CSV 檔案的資料夾路徑。
    - pattern: 檔名匹配規則（例如 'B6031600_*.csv'）。
    - u_col: 目標電壓欄位名稱（例如 'U1'）。
    - i_col: 輸入電流欄位名稱（例如 'I1'）。
    - event_col: 標記事件編號的欄位名稱（例如 'EventNo'）。
    - downsample_factor: 降採樣倍率（預設為 1，不降採樣；大於 1 時使用 FIR 抗混疊濾波降採樣）。
    
    回傳:
    - U_list: 每個事件的電壓序列列表 (長度為 N_episodes)。
    - I_list: 每個事件的電流序列列表 (長度為 N_episodes)。
    - episode_ids: 每個事件的唯一識別識別碼。
    - lengths: 每個事件的採樣點長度列表。
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
            u = group[u_col].values.astype(np.float64)  # decimate 要求 float64
            i = group[i_col].values.astype(np.float64)

            # 使用 FIR 零相位濾波降採樣，避免高頻混疊（Anti-Aliasing）
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
    """
    自定義 PyTorch 資料集類別：保留每個事件原本的時序長度（變長序列）。
    維度增加一維特徵維度: (T_i, 1)。
    """
    def __init__(self, I_list, U_list):
        self.I = [torch.tensor(i, dtype=torch.float32).unsqueeze(-1) for i in I_list]  # (T_i, 1)
        self.U = [torch.tensor(u, dtype=torch.float32).unsqueeze(-1) for u in U_list]

    def __len__(self):
        return len(self.I)

    def __getitem__(self, idx):
        return self.I[idx], self.U[idx]


def collate_variable_length(batch):
    """
    DataLoader 的整理函數（collate_fn）：動態填充變長序列並建立遮罩。
    
    回傳:
    - i_padded: (batch, T_max, 1)
    - u_padded: (batch, T_max, 1)
    - lengths: (batch,)
    - mask: (batch, T_max, 1)，真實點為 1.0，補零點為 0.0
    """
    i_seqs, u_seqs = zip(*batch)
    lengths = torch.tensor([len(s) for s in i_seqs], dtype=torch.long)

    # 填充至該 Batch 最大長度 T_max
    i_padded = pad_sequence(i_seqs, batch_first=True, padding_value=0.0)
    u_padded = pad_sequence(u_seqs, batch_first=True, padding_value=0.0)

    # 建立二元遮罩
    T_max = i_padded.size(1)
    mask = (torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)

    return i_padded, u_padded, lengths, mask
