"""Nạp, chia validation và chuẩn hoá dữ liệu Forest CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_path = Path(processed_dir)
    train_path = processed_path / "train.npz"
    eval_path = processed_path / "eval.npz"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy train.npz/eval.npz trong {processed_path.resolve()}. "
            "Hãy chạy `python scripts/split_data.py` từ thư mục gốc repo."
        )

    with np.load(train_path, allow_pickle=False) as train_data:
        X_train = train_data["X"]
        y_train = train_data["y"]
    with np.load(eval_path, allow_pickle=False) as eval_data:
        X_eval = eval_data["X"]
        y_eval = eval_data["y"]
        eval_row_id = eval_data["row_id"]

    _validate_arrays(X_train, y_train, "train")
    _validate_arrays(X_eval, y_eval, "eval")
    if eval_row_id.dtype != np.int64 or eval_row_id.shape != y_eval.shape:
        raise ValueError("eval row_id phải có dtype int64 và shape (N,).")
    if np.unique(eval_row_id).size != eval_row_id.size:
        raise ValueError("eval row_id chứa giá trị trùng lặp.")

    return X_train, y_train, X_eval, y_eval, eval_row_id


def _validate_arrays(X: np.ndarray, y: np.ndarray, split_name: str) -> None:
    """Kiểm tra sớm định dạng dữ liệu để lỗi không lan vào vòng huấn luyện."""
    if X.ndim != 2 or X.shape[1] != 54:
        raise ValueError(f"{split_name}: X phải có shape (N, 54), nhận {X.shape}.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"{split_name}: y phải có shape (N,) và cùng số mẫu với X.")
    if X.dtype != np.float32 or y.dtype != np.int64:
        raise TypeError(
            f"{split_name}: cần X=float32, y=int64; nhận X={X.dtype}, y={y.dtype}."
        )
    if y.size == 0 or int(y.min()) < 0 or int(y.max()) > 6:
        raise ValueError(f"{split_name}: nhãn phải nằm trong 0..6.")


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm trong khoảng (0, 1).")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu.")
    return train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có shape (N, >= {N_NUMERIC}).")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Mean/std chứa NaN hoặc inf.")
    # Không cột nào của CoverType có std=0, nhưng guard này giúp hàm an toàn hơn.
    std = np.where(std == 0, np.float32(1.0), std).astype(np.float32)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X phải có shape (N, >= {N_NUMERIC}).")
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(f"mean/std phải có shape ({N_NUMERIC},).")
    if np.any(std <= 0):
        raise ValueError("Mọi phần tử std phải dương.")

    result = np.array(X, dtype=np.float32, copy=True)
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC] - mean) / std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train, y_train, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, X_val, y_tr, y_val = make_val_split(
        X_train, y_train, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    torch_device = torch.device(device)

    def as_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=torch_device)

    data = {
        "X_tr": as_tensor(X_tr, torch.float32),
        "y_tr": as_tensor(y_tr, torch.int64),
        "X_val": as_tensor(X_val, torch.float32),
        "y_val": as_tensor(y_val, torch.int64),
        "X_eval": as_tensor(X_eval, torch.float32),
        "y_eval": as_tensor(y_eval, torch.int64),
        "eval_row_id": eval_row_id.copy(),
        "mean": mean,
        "std": std,
    }

    majority_class = int(np.bincount(y_tr, minlength=7).argmax())
    majority_acc = float(np.mean(y_val == majority_class))
    print(
        f"train={len(y_tr):,}, val={len(y_val):,}, eval={len(y_eval):,}; "
        f"device={torch_device}"
    )
    print(
        f"Luôn đoán lớp đa số {majority_class}: val accuracy={majority_acc:.4f}"
    )
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương.")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu.")

    if shuffle:
        indices = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        indices = torch.arange(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]
