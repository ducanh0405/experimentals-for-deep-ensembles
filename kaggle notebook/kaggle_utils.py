"""
Helpers for Kaggle/local runs: paths, optional Weights & Biases, ResNet module download.

Kaggle thực tế:
- Đặt `kaggle_utils.py` cùng thư mục với các notebook `.py` trong `/kaggle/working` (hoặc gắn Dataset chứa file này).
- Tải `resnet_cifar10.py`: bật Internet (Settings → Internet) nếu dùng `ensure_resnet_cifar10_module`.
- GPU: chọn accelerator GPU; notebook 0 gọi `nvidia-smi` an toàn nếu không có GPU.
"""

from __future__ import annotations

import glob
import os
import sys
import urllib.request


IS_KAGGLE = os.path.isdir("/kaggle/working")
WORKING_DIR = "/kaggle/working" if IS_KAGGLE else os.path.abspath(".")

# Giúp `from kaggle_utils import ...` khi chạy notebook/script từ thư mục khác cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, WORKING_DIR):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def tf_autotune():
    import tensorflow as tf

    if hasattr(tf.data, "AUTOTUNE"):
        return tf.data.AUTOTUNE
    experimental = getattr(tf.data, "experimental", None)
    if experimental is not None and hasattr(experimental, "AUTOTUNE"):
        return experimental.AUTOTUNE
    return 4


def get_weights_dir(target_dir: str) -> str:
    w = os.path.join(WORKING_DIR, target_dir)
    if os.path.isdir(w):
        return w
    input_paths = glob.glob(f"/kaggle/input/*/{target_dir}")
    if input_paths:
        return input_paths[0]
    input_root = glob.glob(f"/kaggle/input/{target_dir}")
    if input_root:
        return input_root[0]
    local = os.path.join("." if not IS_KAGGLE else WORKING_DIR, target_dir)
    return os.path.abspath(local)


def output_path(filename: str) -> str:
    return os.path.join(WORKING_DIR, filename)


def load_keras_model(filepath: str):
    """
    Load .h5 để inference / ensemble. compile=False tránh lỗi tương thích optimizer giữa các bản TF trên Kaggle.
    """
    import tensorflow as tf

    return tf.keras.models.load_model(filepath, compile=False)


def wandb_login_optional() -> bool:
    """
    Đăng nhập W&B nếu có WANDB_API_KEY (Kaggle: Secrets → Add to environment).
    Trả về True nếu nên dùng wandb.init / log / WandbCallback.
    """
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        print(
            "WANDB_API_KEY chưa đặt — chạy không log W&B. "
            "Trên Kaggle: Add-ons → Secrets → WANDB_API_KEY → Attach to notebook."
        )
        return False
    try:
        import wandb
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "wandb", "--quiet"]
        )
        import wandb

    wandb.login(key=key)
    return True


def ensure_resnet_cifar10_module(filename: str = "resnet_cifar10.py") -> None:
    if os.path.isfile(filename):
        return
    url = (
        "https://raw.githubusercontent.com/GoogleCloudPlatform/"
        "keras-idiomatic-programmer/master/zoo/resnet/resnet_cifar10.py"
    )
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filename)
    except OSError as e:
        raise RuntimeError(
            "Không tải được resnet_cifar10.py. Trên Kaggle: Settings → Internet → On, "
            "hoặc tự thêm file vào working / Dataset rồi chạy lại."
        ) from e


def pip_install_quiet(package: str) -> None:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_umap() -> None:
    try:
        import umap  # noqa: F401
    except ImportError:
        pip_install_quiet("umap-learn")


def display_df(df):
    try:
        from IPython.display import display

        display(df)
    except ImportError:
        print(df.to_string())
