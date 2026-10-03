"""model.py — MLP cho bài toán 7 lớp, shape cố định (README mục 3, GUIDE "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn. Mọi nn.Linear có bias. Không BatchNorm, không residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers, d = [], in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))   # sau ReLU của lớp ẩn
            d = h
        layers.append(nn.Linear(d, num_classes))      # lớp ra: logit thô
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo W của MỌI nn.Linear theo `init`; bias luôn = 0 (trừ "default": giữ nguyên mặc định của PyTorch).

    "xavier" ở đây là nn.init.xavier_normal_ (Var = 2/(n_in+n_out)), KHÔNG phải Var = 1/n_in của slide.
    """
    if init == "default":
        return
    fns = {
        "zeros": nn.init.zeros_,
        "normal": lambda w: nn.init.normal_(w, std=0.01),
        "xavier": nn.init.xavier_normal_,
        "he": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),   # Var = 2/n_in (fan_in)
    }
    if init not in fns:
        raise ValueError(f"init phải thuộc {list(fns) + ['default']}, nhận {init!r}")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            fns[init](m.weight)
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt SAU MỖI nn.Linear (pre-activation; lớp cuối = logit) trên một lô x, ở eval mode."""
    was_training = model.training
    model.eval()
    h, stds = x, []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(h.std().item())
    model.train(was_training)
    return stds
