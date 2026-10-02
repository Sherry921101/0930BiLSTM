# V-I Curve Modeling (電壓-電流曲線動態代理模型)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基於雙向長短期記憶神經網路（Bidirectional LSTM, BiLSTM）的電力系統動態電壓-電流特性曲線代理模型（Surrogate Model）。
本專案專為變長事件序列（Variable-length Episodes）設計，能依據量測電流序列（$I$）精準重構並預測對應的電壓響應（$U$）。

---

## 📌 核心特色 (Key Features)

- **變長序列高效處理**：利用 PyTorch 的 `pack_padded_sequence` 與 `pad_packed_sequence`，避免填充值（Padding）干擾 LSTM 內部隱藏狀態，兼顧計算效率與時序真實性。
- **動態遮罩損失函數 (Masked MSE)**：在計算訓練與驗證損失時，精確遮蔽填充區域，只對真實量測時步計算誤差。
- **抗混疊降採樣 (Anti-aliasing Decimation)**：內建 `scipy.signal.decimate` FIR 低通抗混疊濾波降採樣，可平滑處理高頻量測資料。
- **基準模型對照 (Baseline Benchmark)**：自動擬合無記憶效應的靜態線性回歸（Linear Regression）作為 Baseline，量化評估 BiLSTM 對動態記憶效應的提升幅度。
- **模組化命令列參數 (CLI Support)**：完整支援 `argparse`，可彈性調整資料路徑、欄位名稱、超參數與輸出路徑。
- **豐富的視覺化評估**：自動輸出代表性（典型樣本與極端案例）預測曲線圖與逐事件評估報表。

---

## 📁 專案結構 (Project Structure)

```text
vi_curve_modeling/
├── BiLSTM.py               # 執行入口主程式 (協調資料載入、訓練與評估主流程)
├── src/                    # 核心功能模組套件
│   ├── __init__.py         # 模組統一匯出入口
│   ├── data.py             # 資料載入、抗混疊降採樣、Dataset 與動態批次填充 (collate_fn)
│   ├── model.py            # BiLSTMSurrogate 雙向神經網路架構
│   ├── trainer.py          # 遮罩均方誤差 (Masked MSE)、訓練迴圈與推論引擎
│   ├── metrics.py          # 工程評估指標計算 (RMSE, MAE, R², Fit Acc, 2% Tol)
│   └── visualization.py    # 動態暫態特徵鎖定演算法與三合一高精度繪圖
├── requirements.txt        # 專案依賴套件清單
├── .gitignore              # Git 忽略設定檔（排除大型資料集、虛擬環境與快取）
├── .gitattributes          # Git 屬性設定
├── README.md               # 專案說明文件
└── LICENSE                 # MIT 開源授權條款
```

---

## 🛠️ 環境安裝 (Installation)

建議使用 Python 3.9 以上之虛擬環境：

```bash
# 1. 複製本專案
git clone https://github.com/Sherry921101/vi_curve_modeling.git
cd vi_curve_modeling

# 2. 建立並啟動虛擬環境 (以 Windows PowerShell 為例)
python -m venv venv
.\venv\Scripts\Activate.ps1

# (若在 Linux / macOS)
# python3 -m venv venv
# source venv/bin/activate

# 3. 安裝依賴套件
pip install -r requirements.txt
```

---

## 📊 資料集準備 (Dataset Preparation)

> **說明**：為保持 GitHub 倉庫輕量化，原始量測 CSV 資料（大檔）已加入 `.gitignore` 排除清單，並未上傳至遠端倉庫。

請將您的量測 CSV 檔案放置於專案根目錄下的 `data/` 資料夾（或自訂資料夾），程式會依據 `--file_pattern` 自動讀取。

### CSV 格式需求：
每個 CSV 檔案應包含以下欄位：
- `EventNo`：事件識別編號（每個事件視為一個獨立的變長 Episode）。
- `I1`：輸入特徵（量測電流）。
- `U1`：預測目標（量測電壓）。

範例結構：
```csv
EventNo,U1,I1
1,114.2,5.12
1,114.1,5.15
...
2,113.8,4.98
```

---

## 🚀 快速開始 (Quick Start)

### 1. 預設執行 (自動偵測資料夾)
若本地已有資料資料夾（如 `B6031600_event_all` 或 `data/`）：
```bash
python BiLSTM.py
```

### 2. 自訂參數執行
可透過命令列引數自訂執行設定：
```bash
python BiLSTM.py \
  --data_dir ./data \
  --file_pattern "B6031600_*.csv" \
  --epochs 30 \
  --batch_size 32 \
  --hidden_size 64 \
  --lr 0.001 \
  --output_dir results
```

### 3. 查看完整引數清單
```bash
python BiLSTM.py --help
```

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--data_dir` | `None` (自動偵測) | 資料集所在資料夾路徑 |
| `--file_pattern` | `B6031600_*.csv` | 檔案名稱匹配樣式 (Glob) |
| `--u_col` | `U1` | 電壓目標欄位名稱 |
| `--i_col` | `I1` | 電流輸入欄位名稱 |
| `--event_col` | `EventNo` | 事件編號欄位名稱 |
| `--downsample` | `1` | 降採樣倍率（1 為不降採樣） |
| `--epochs` | `20` | 訓練輪數 |
| `--batch_size` | `32` | 批次大小 |
| `--hidden_size`| `32` | BiLSTM 隱藏層單元數 |
| `--num_layers` | `1` | BiLSTM 堆疊層數 |
| `--lr` | `0.001` | Adam 優化器學習率 |
| `--output_dir` | `results` | 成果圖表與評估結果儲存路徑 |
| `--zoom_window`| `150` | 高精度放大對比圖所展示的動態瞬態採樣點數 |
| `--baseline_type` | `xgboost` | 基準對照組架構：`xgboost` (XGBoost 梯度提升樹)、`rf` (隨機森林) 或 `linear` (線性回歸) |
| `--seed` | `42` | 隨機種子 (Reproducibility) |
| `--eval_only` | `False` | 略過訓練，直接載入已訓練權重進行評估與繪圖 |

---

## 📈 輸出成果 (Outputs)

訓練與測試完成後，成果將自動儲存於 `--output_dir`（預設為 `results/`）：

1. **`vi_bilstm_model.pt`**：訓練完畢之 PyTorch 模型權重存檔。
2. **`vi_bilstm_test_results.csv`**：測試集中各事件的詳細評估指標，包含：
   - **Fit Accuracy（擬合準確率 %）**：基於 NRMSE，評估全動態幅值範圍內的波形吻合率
   - **$R^2$ 判定係數（Goodness of Fit）**：衡量模型對真實電壓波動趨勢的解釋能力
   - **RMSE（均方根誤差）**：平方放大加總後開根號，著重處罰暫態跳變時的極值大誤差
   - **MAE（平均絕對誤差）**：各時刻絕對偏差的算術平均，代表直觀物理誤差
   - **MSE（均方誤差）**：模型反向傳播訓練損失之基準
3. **`vi_bilstm_example_typical_high_precision.png`** & **`vi_bilstm_example_worst_case_high_precision.png`**：
   - **高精度多面板放大對比圖（300 DPI）**：
     - **Panel 1 (全域波形)**：完整 Episode 波形比對、標註瞬態高動態選取區，並附上 Accuracy / $R^2$ / RMSE 性能資訊卡。
     - **Panel 2 (局部高精度放大)**：鎖定訊號變化最劇烈的瞬態/躍階區段，以高解析度點標記（Markers）展示點對點緊密貼合度。
     - **Panel 3 (瞬時絕對誤差殘差)**：直觀對比 BiLSTM 誤差殘差 vs 靜態線性模型誤差。
4. **`vi_bilstm_example_typical.png`** & **`vi_bilstm_example_worst_case.png`**：標準全域曲線對照圖。

---

## 📄 授權條款 (License)

本專案採用 [MIT License](LICENSE) 開源授權條款。
