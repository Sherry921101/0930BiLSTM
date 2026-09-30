import sys
import time
import torch
from tqdm import tqdm


def masked_mse(pred, target, mask):
    """
    遮罩均方誤差（Masked MSE）：
    只計算真實有效時間步的均方誤差，完全忽略 padding 補零位置，防止梯度受到虛假 0 值的污染。
    
    公式:
    MSE = sum((pred - target)^2 * mask) / sum(mask)
    """
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1.0)


def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3):
    """
    模型訓練迴圈：
    - 採用 Adam 優化器
    - 支援巢狀進度條（外層 Epoch 條 + 內層 Batch 條）
    - 每個 Epoch 計算真實有效點的 Train MSE 與 Validation MSE
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training {model.__class__.__name__} (epochs={epochs}, device={device}) ---", flush=True)
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_bar:
        # ---- 訓練階段 (Train) ----
        model.train()
        running, running_count = 0.0, 0
        t0 = time.time()
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{epochs} [train]",
            unit="batch",
            leave=False,
            position=1,
        )
        for bi, bu, lengths, mask in batch_bar:
            bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(bi, lengths)
            loss = masked_mse(pred, bu, mask)
            loss.backward()
            optimizer.step()
            running += loss.item() * bi.size(0)
            running_count += bi.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.5f}", seq_len=bi.size(1))
        train_loss = running / running_count
        batch_bar.close()

        # ---- 驗證階段 (Validate) ----
        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1}/{epochs} [val]  ",
                unit="batch",
                leave=False,
                position=1,
            )
            for bi, bu, lengths, mask in val_bar:
                bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
                pred = model(bi, lengths)
                loss = masked_mse(pred, bu, mask)
                val_running += loss.item() * bi.size(0)
                val_count += bi.size(0)
                val_bar.set_postfix(val_loss=f"{loss.item():.5f}")
            val_bar.close()
        val_loss = val_running / val_count

        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            sec=f"{elapsed:.1f}s",
        )
        tqdm.write(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"Train MSE: {train_loss:.5f}  Val MSE: {val_loss:.5f}  "
            f"({elapsed:.1f}s)",
            file=sys.stdout,
        )

    epoch_bar.close()
    return model


def predict_per_episode(model, loader, device):
    """
    依事件逐一推論並截斷補零：
    推論完成後，根據每個事件的真實長度 lengths[b] 將多餘的 Padding 切除，還原原始時序長度。
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for bi, _, lengths, _ in tqdm(loader, desc="Predicting", unit="batch"):
            bi = bi.to(device)
            out = model(bi, lengths).cpu().numpy()
            for b in range(out.shape[0]):
                preds.append(out[b, :lengths[b], :])
    return preds
