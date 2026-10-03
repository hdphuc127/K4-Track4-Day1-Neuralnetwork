"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_SUBSET = 50_000   # train_loss được đo (eval mode) trên một tập con CỐ ĐỊNH của train

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine"
    seed=1,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = TB cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0. cm: hàng = thật, cột = dự đoán."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    p = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax logits, ở eval mode (không dropout)."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str):
    """"ce": F.cross_entropy(logit thô, nhãn int64) — trung bình trên lô.
    "mse": nn.MSELoss mặc định = trung bình bình phương sai số trên MỌI phần tử (B×7) giữa logit và one-hot
    (không có hệ số 1/2, không softmax)."""
    if loss_name == "ce":
        return F.cross_entropy(logits.float(), y)
    if loss_name == "mse":
        return F.mse_loss(logits.float(), F.one_hot(y, N_CLASSES).float())
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở eval() + no_grad. Loss = trung bình đúng trên N mẫu (cộng theo lô rồi chia N)."""
    model.eval()
    total, cm = 0.0, torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.long, device=X.device)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name) * len(xb)   # trung bình lô × kích thước lô
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(1), minlength=N_CLASSES ** 2)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=float(total) / len(X), acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def _nan_to_none(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện MỘT cấu hình; trả về {"cfg", "history", "summary", "best_state"} (xem cột experiments.xlsx).

    Chỉ dùng train/val. KHÔNG đụng X_eval: chọn epoch tốt nhất bằng val_loss thấp nhất (giống dừng sớm).
    Đo gì: train_loss (eval mode, tập con cố định của train), val_loss/acc/macro-F1, grad_norm TRƯỚC clip
    (trung bình/lớn nhất/trung vị theo epoch + tỉ lệ bước bị clip), thời gian huấn luyện mỗi epoch, bộ nhớ GPU cực đại,
    loss bước 0 và độ lệch chuẩn kích hoạt ở bước 0, cờ diverged (loss NaN/inf -> dừng sớm).
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "cfg['lr'] chưa được đặt"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    use_cuda = device.type == "cuda"
    prec = cfg["precision"]
    if prec == "fp16" and not use_cuda:
        raise RuntimeError("fp16 + GradScaler cần GPU CUDA")
    if prec == "bf16" and use_cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU này không hỗ trợ bf16")

    # ---- 0. model, optimizer
    set_seed(cfg["seed"])
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], count_params(model)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    sched = build_scheduler(opt, cfg["scheduler"], cfg["epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda") if prec == "fp16" else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(prec)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])       # xáo lô: phụ thuộc seed của thí nghiệm
    sub = torch.randperm(len(X_tr), generator=torch.Generator(device=device).manual_seed(0), device=device)[:TRAIN_SUBSET]
    X_sub, y_sub = X_tr[sub], y_tr[sub]                                  # tập con train CỐ ĐỊNH để đo train_loss

    # ---- 1. loss bước 0 (trước bước cập nhật đầu tiên) và độ lệch chuẩn kích hoạt
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]
    act_std = activation_stats(model, X_val[:4096])
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "grad_norm_max",
                            "grad_norm_p50", "clip_frac", "epoch_time_s")}
    best = dict(val_loss=float("inf"), epoch=None, state=None)
    diverged, skipped = False, 0

    # ---- 2. huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gns = []
        for step, (xb, yb) in enumerate(iterate_batches(X_tr, y_tr, cfg["batch"], gen)):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):   # chỉ bọc forward + loss
                loss = compute_loss(model(xb), yb, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)               # luôn unscale trước khi đo/clip để grad_norm đúng thang
                gns.append(clip_gradients(model.parameters(), cfg["clip_norm"]))
                scaler.step(opt)                   # tự bỏ qua bước nếu gradient inf/NaN
                scaler.update()
            else:
                loss.backward()
                gns.append(clip_gradients(model.parameters(), cfg["clip_norm"]))   # chuẩn TRƯỚC khi cắt
                opt.step()
            if sched is not None:
                sched.step()
            if step % 50 == 0 and not torch.isfinite(loss):    # đồng bộ hiếm để không làm chậm GPU
                diverged = True
                break
        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # ---- cuối epoch: đo ở eval mode
        gn = torch.stack(gns).float()
        finite = torch.isfinite(gn)
        skipped += int((~finite).sum())            # fp16: bước bị GradScaler bỏ qua vì gradient tràn số
        gn = gn[finite] if finite.any() else gn.new_full((1,), float("nan"))
        trn = evaluate(model, X_sub, y_sub, cfg["loss"])
        val = evaluate(model, X_val, y_val, cfg["loss"])
        diverged = diverged or not math.isfinite(val["loss"])
        for k, v in dict(epoch=epoch, train_loss=trn["loss"], val_loss=val["loss"], val_acc=val["acc"],
                         val_macro_f1=val["macro_f1"], grad_norm=float(gn.mean()), grad_norm_max=float(gn.max()),
                         grad_norm_p50=float(gn.median()),
                         clip_frac=float((gn > cfg["clip_norm"]).float().mean()) if cfg["clip_norm"] else 0.0,
                         epoch_time_s=epoch_time).items():
            hist[k].append(v)
        if val["loss"] < best["val_loss"]:
            best.update(val_loss=val["loss"], epoch=epoch, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if diverged:
            break

    # ---- 3. tóm tắt tại best_epoch
    i = None if best["epoch"] is None else best["epoch"] - 1
    at = lambda k: None if i is None else _nan_to_none(hist[k][i])
    summary = dict(
        step0_loss=step0_loss, best_val_loss=at("val_loss"), best_epoch=best["epoch"],
        final_train_loss=_nan_to_none(hist["train_loss"][-1]), final_val_loss=_nan_to_none(hist["val_loss"][-1]),
        val_acc=at("val_acc"), val_macro_f1=at("val_macro_f1"),
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
        diverged=diverged,
        # thông tin phụ (không có cột riêng trong bảng; đưa vào notes khi cần)
        act_std_step0=act_std, skipped_steps=skipped,
        clip_frac_mean=float(np.mean(hist["clip_frac"])), n_epochs_run=len(hist["epoch"]),
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def write_predictions(row_id, preds, path: str) -> None:
    """CSV `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng eval, mỗi row_id đúng một lần."""
    import pandas as pd
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) == len(set(row_id.tolist())) and preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho baseline và cấu hình cuối cùng: nạp best_state (epoch val_loss thấp nhất), dự đoán TOÀN BỘ eval
    (fp32, eval mode) rồi ghi file dự đoán. Điểm do `scripts/evaluate.py` tính, không tính ở đây."""
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    write_predictions(data["eval_row_id"], predict(model, data["X_eval"]).cpu().numpy(), pred_path)
