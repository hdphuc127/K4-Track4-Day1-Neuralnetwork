"""plots.py — ảnh biểu đồ là sản phẩm nộp: mỗi thí nghiệm một ảnh figures/<exp_id>.png,
mỗi nhóm thí nghiệm nên có thêm một ảnh chồng figures/compare_<nhóm>.png.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def _cfg_text(cfg: dict) -> str:
    clip = cfg.get("clip_norm")
    return (f"{cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | {cfg['loss']} | batch={cfg['batch']} | "
            f"hidden={'-'.join(map(str, cfg['hidden']))} | drop={cfg['dropout']:g} | clip={format(clip, 'g') if clip else 'none'} | "
            f"{cfg['precision']} | init={cfg['init']} | seed={cfg['seed']}")


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm (đo TRƯỚC clip)."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    ax[0].plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax[0].plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax[0].set(xlabel="epoch", ylabel=f"loss ({cfg['loss'].upper()})", title="Train / val loss")

    ax[1].plot(ep, h["val_acc"], "o-", ms=3, color="tab:green", label="val accuracy")
    ax[1].plot(ep, h["val_macro_f1"], "s-", ms=3, color="tab:red", label="val macro-F1")
    ax[1].set(xlabel="epoch", ylabel="điểm (0–1)", title="Val accuracy / macro-F1", ylim=(0, 1))

    ax[2].plot(ep, h["grad_norm"], "o-", ms=3, color="tab:purple", label="trung bình / epoch")
    ax[2].plot(ep, h["grad_norm_max"], "^--", ms=3, color="tab:orange", label="lớn nhất / epoch (gai)")
    if cfg.get("clip_norm"):
        ax[2].axhline(cfg["clip_norm"], color="k", ls=":", label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    ax[2].set(xlabel="epoch", ylabel="‖g‖₂ toàn cục (trước clip)", title="Grad norm")
    try:
        ax[2].set_yscale("log")
    except ValueError:
        pass

    be = s.get("best_epoch")
    for a in ax:
        if be:
            a.axvline(be, color="gray", ls="--", lw=1, label=f"best epoch = {be}")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    flag = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg['exp_id']}{flag}\n{_cfg_text(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm, chú thích bằng exp_id."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        h = r["history"]
        ax.plot(h["epoch"], h[metric], "o-", ms=3, label=r["cfg"]["exp_id"])
    ax.set(xlabel="epoch", ylabel=metric, title=title or f"So sánh {metric}")
    if metric in ("grad_norm", "grad_norm_max"):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
