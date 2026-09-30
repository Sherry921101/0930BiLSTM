import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMSurrogate(nn.Module):
    """
    雙向 LSTM 代理模型（Bidirectional LSTM Surrogate Model）：
    以動態電流輸入序列 I(t) 預測響應電壓序列 U(t)。
    
    雙向機制能同時捕捉事件前後時間脈絡，比單向模型更能精準擬合暫態震盪與遲滯效應。
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,       # 輸入特徵維度 (電流=1)
            hidden_size=hidden_size,     # 隱藏層特徵維度
            num_layers=num_layers,       # 堆疊 LSTM 層數
            batch_first=True,            # (batch, seq_len, feature)
            bidirectional=True,          # 啟用雙向 LSTM
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 輸出層全連接網路：雙向隱藏狀態 (hidden_size * 2) 映射至電壓 (1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, i_seq, lengths):
        """
        前向傳播：
        - i_seq: (batch, T_max, 1)
        - lengths: (batch,) 實際長度
        """
        # 使用 pack_padded_sequence 壓縮填充時間步，避免 padding 影響狀態計算
        packed = pack_padded_sequence(i_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        
        # pad_packed_sequence: 解包還原為張量 (batch, T_max, hidden_size * 2)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=i_seq.size(1))
        
        # 全連接映射
        u_pred = self.fc(out)  # (batch, T_max, 1)
        return u_pred
