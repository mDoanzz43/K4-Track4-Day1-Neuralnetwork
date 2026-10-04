"""Pipeline huấn luyện, đánh giá và xuất dự đoán dùng chung cho mọi run.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time
from copy import deepcopy
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # Phải chọn bằng validation trước khi chạy.
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    train_eval_size=50_000,
    progress=True,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError(f"cm phải có shape (7, 7), nhận {cm.shape}.")
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải dương.")
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start:start + batch_size])
        outputs.append(logits.argmax(dim=1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=torch.int64, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X/y phải cùng số mẫu và không được rỗng.")
    if loss_name not in {"ce", "mse"}:
        raise ValueError("loss_name phải là 'ce' hoặc 'mse'.")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    confusion = torch.zeros((7, 7), dtype=torch.int64, device=X.device)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
        else:
            targets = F.one_hot(yb, num_classes=7).to(dtype=logits.dtype)
            total_loss += float(F.mse_loss(logits, targets, reduction="sum").item())
        pred = logits.argmax(dim=1)
        total_correct += int((pred == yb).sum().item())
        confusion += torch.bincount(yb * 7 + pred, minlength=49).reshape(7, 7)
    denominator = len(y) if loss_name == "ce" else len(y) * 7
    cm = confusion.cpu().numpy()
    return {
        "loss": total_loss / denominator,
        "acc": total_correct / len(y),
        "macro_f1": macro_f1_from_confusion(cm),
        "confusion_matrix": cm.tolist(),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        targets = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, targets)
    raise ValueError(f"Loss chưa hỗ trợ: {loss_name!r}.")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    config = {**DEFAULT_CFG, **cfg}
    if config["lr"] is None or float(config["lr"]) <= 0:
        raise ValueError("cfg['lr'] phải là số dương và phải được chọn bằng validation.")
    if config["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là fp32, fp16 hoặc bf16.")

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if config["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("FP16/BF16 trong lab này yêu cầu CUDA.")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU hiện tại không hỗ trợ BF16 bằng phần cứng.")

    set_seed(int(config["seed"]))
    model = MLP(hidden=tuple(config["hidden"]), dropout=float(config["dropout"]),
                init=config["init"]).to(device)
    hidden_key = tuple(config["hidden"])
    if hidden_key in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden_key]
    optimizer = build_optimizer(
        config["optimizer"], model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
        momentum=float(config.get("momentum", 0.9)),
        betas=tuple(config.get("betas", (0.9, 0.999))), eps=float(config.get("eps", 1e-8)),
    )
    use_fp16 = config["precision"] == "fp16"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0 = evaluate(model, X_val, y_val, config["loss"])["loss"]
    history = {key: [] for key in (
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s"
    )}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = None
    diverged = False
    train_eval_size = min(int(config.get("train_eval_size", 50_000)), len(X_tr))
    X_train_eval, y_train_eval = X_tr[:train_eval_size], y_tr[:train_eval_size]

    epoch_iterator = tqdm(
        range(1, int(config["epochs"]) + 1),
        desc=config["exp_id"], unit="epoch", disable=not bool(config.get("progress", True)),
        dynamic_ncols=True,
    )
    for epoch in epoch_iterator:
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        grad_norm_sum, steps = 0.0, 0
        for xb, yb in iterate_batches(X_tr, y_tr, int(config["batch"]), generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=config["precision"] != "fp32"):
                logits = model(xb)
                loss = compute_loss(logits, yb, config["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_gradients(model.parameters(), config.get("clip_norm"))
            if not np.isfinite(grad_norm):
                diverged = True
                break
            grad_norm_sum += grad_norm
            steps += 1
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - started
        if diverged:
            epoch_iterator.set_postfix_str("diverged")
            break

        train_metrics = evaluate(model, X_train_eval, y_train_eval, config["loss"])
        val_metrics = evaluate(model, X_val, y_val, config["loss"])
        values = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "grad_norm": grad_norm_sum / max(steps, 1), "epoch_time_s": epoch_time,
        }
        for key, value in values.items():
            history[key].append(value)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        epoch_iterator.set_postfix(
            loss=f"{val_metrics['loss']:.4f}", f1=f"{val_metrics['macro_f1']:.4f}",
            grad=f"{values['grad_norm']:.2f}", sec=f"{epoch_time:.1f}", refresh=True,
        )

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    if best_state is None:
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        best_metrics = {"loss": float("nan"), "acc": 0.0, "macro_f1": 0.0}
    summary = {
        "step0_loss": float(step0), "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        "val_acc": float(best_metrics["acc"]), "val_macro_f1": float(best_metrics["macro_f1"]),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": float(peak_mem_mb), "diverged": bool(diverged),
    }
    return {"cfg": config, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id/preds phải là vector cùng độ dài.")
    if np.unique(row_id).size != len(row_id):
        raise ValueError("row_id bị trùng.")
    if not np.issubdtype(preds.dtype, np.integer) or np.any((preds < 0) | (preds > 6)):
        raise ValueError("pred phải là số nguyên trong 0..6.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_id.tolist(), preds.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg["dropout"]),
                init=cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    return preds
