"""Vẽ ảnh bắt buộc cho từng run và ảnh so sánh theo nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history, summary = result["cfg"], result["history"], result["summary"]
    epochs = history["epoch"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", ms=3, label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", ms=3, label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel=cfg["loss"].upper())
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", ms=3, label="val accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], marker="o", ms=3, label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()

    axes[2].plot(epochs, history["grad_norm"], marker="o", ms=3, color="tab:red", label="before clip")
    axes[2].set(title="Global gradient norm", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()

    best_epoch = summary.get("best_epoch")
    if best_epoch:
        for axis in axes:
            axis.axvline(best_epoch, color="black", ls="--", alpha=.35, label="best epoch")
    for axis in axes:
        axis.grid(alpha=.25)
    main = (
        f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']:g} | "
        f"batch={cfg['batch']} {cfg['precision']}"
    )
    fig.suptitle(main, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results không được rỗng.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"Không có metric {metric!r} trong history.")
        ax.plot(history["epoch"], history[metric], marker="o", ms=3,
                label=result["cfg"]["exp_id"])
    ax.set(title=title or f"Compare {metric}", xlabel="Epoch", ylabel=metric)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
