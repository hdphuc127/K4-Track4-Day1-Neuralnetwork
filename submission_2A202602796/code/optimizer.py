"""optimizer.py — chọn bộ tối ưu (torch.optim) và cắt gradient, để mọi thí nghiệm dùng chung một đường.

    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr*wd*w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về torch.optim.Optimizer. weight_decay của Adam là L2 trộn vào gradient; của AdamW là suy giảm tách riêng."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch lr. Trả về None nếu name là None; "cosine" -> CosineAnnealingLR theo bước."""
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"scheduler không hỗ trợ: {name!r}")


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    """Cắt gradient theo chuẩn L2 toàn cục; TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt (tensor 0-d, chưa đồng bộ GPU).

    Trả về tensor thay vì float để vòng huấn luyện không phải đồng bộ GPU↔CPU ở mỗi bước; train.py gom lại cuối epoch.
    max_norm=None: chỉ đo (clip_grad_norm_ với max_norm=inf không đổi gradient).
    """
    return torch.nn.utils.clip_grad_norm_(params, math.inf if max_norm is None else max_norm)
