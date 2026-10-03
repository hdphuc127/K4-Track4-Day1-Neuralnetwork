"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu:
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ .npz. Trả về X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_tr, y_tr = tr["X"], tr["y"]
    X_ev, y_ev, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_tr, y_tr), (X_ev, y_ev)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64 and y.min() == 0 and y.max() == 6
    assert len(row_id) == len(X_ev)
    return X_tr, y_tr, X_ev, y_ev, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về X_tr, y_tr, X_val, y_val."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean/std của N_NUMERIC cột đầu, CHỈ trên phần train (sau khi tách val).

    Tính trên val/eval (hoặc toàn bộ dữ liệu) là rò rỉ thông tin của tập chưa được phép "nhìn thấy" vào mô hình.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    return num.mean(0), num.std(0)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / np.where(std > 0, std, 1.0)  # std = 0 -> chỉ trừ mean
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict: X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensor trên device, y int64), eval_row_id (numpy),
    cùng mean/std đã dùng để chuẩn hoá và majority_acc_val.
    """
    X_full, y_full, X_ev, y_ev, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_ev = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_ev))

    t = lambda a: torch.as_tensor(a, device=device)
    data = dict(X_tr=t(X_tr), y_tr=t(y_tr), X_val=t(X_val), y_val=t(y_val), X_eval=t(X_ev), y_eval=t(y_ev),
                eval_row_id=row_id, mean=mean, std=std)
    maj = int(np.bincount(y_tr).argmax())
    data["majority_acc_val"] = float((y_val == maj).mean())
    print(f"train {len(X_tr)} | val {len(X_val)} | eval {len(X_ev)}   (device={device})")
    print(f"'luôn đoán lớp đa số' (lớp {maj}) trên val: accuracy = {data['majority_acc_val']:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Lô cuối có thể nhỏ hơn batch_size; ta GIỮ nó (không drop_last) để mỗi epoch thấy đủ mọi mẫu.
    """
    N = len(X)
    perm = torch.randperm(N, generator=generator, device=X.device) if shuffle else torch.arange(N, device=X.device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
