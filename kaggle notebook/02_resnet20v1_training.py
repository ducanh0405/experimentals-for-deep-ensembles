"""
Notebook 2: ResNet20v1 Training
================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Train ResNet20v1 trên CIFAR-10
- Cả 2 phiên bản: KHÔNG data augmentation và CÓ data augmentation
- 5 random initializations mỗi phiên bản
"""

# %%
import os
import sys

if os.path.isdir("/kaggle/working") and "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

import tensorflow as tf
import numpy as np
import time
import json
import subprocess

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt

from kaggle_utils import (
    WORKING_DIR,
    ensure_resnet_cifar10_module,
    tf_autotune,
    wandb_login_optional,
)

WANDB_OK = wandb_login_optional()
wandb = None
if WANDB_OK:

    def _install_wandb():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "wandb", "--quiet"]
        )

    try:
        import wandb as _wandb
    except ImportError:
        _install_wandb()
        import wandb as _wandb
    wandb = _wandb

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# # Download ResNet CIFAR-10 module

# %%
ensure_resnet_cifar10_module()
import resnet_cifar10

# %% [markdown]
# # Tải CIFAR-10

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

BATCH_SIZE = 128
AUTO = tf_autotune()

def normalize(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

def augment(image, label):
    """Data augmentation theo paper."""
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.clip_by_value(image, 0., 1.)
    return image, label

def make_dataset(x, y, shuffle=False, with_augmentation=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.map(normalize, num_parallel_calls=AUTO)
    if with_augmentation:
        ds = ds.map(augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

# %% [markdown]
# # Định nghĩa ResNet20v1

# %%
def get_training_model():
    """ResNet20 (n=2, depth=20)"""
    n = 2
    depth = n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    inputs = Input(shape=(32, 32, 3))
    x = resnet_cifar10.stem(inputs)
    x = resnet_cifar10.learner(x, n_blocks)
    outputs = resnet_cifar10.classifier(x, 10)

    model = Model(inputs, outputs)
    return model

model = get_training_model()
model.summary()
print(f"\nTotal params: {model.count_params()}")

# %% [markdown]
# # LR Schedule

# %%
LR_SCHEDULE = [
    (0, 1.6e-3),
    (9, 8e-4),
    (19, 4e-4),
    (29, 2e-4),
]

def lr_schedule(epoch):
    if (epoch >= 0) and (epoch < 9):
        return LR_SCHEDULE[0][1]
    elif (epoch >= 9) and (epoch < 19):
        return LR_SCHEDULE[1][1]
    elif (epoch >= 19) and (epoch < 29):
        return LR_SCHEDULE[2][1]
    else:
        return LR_SCHEDULE[3][1]

EPOCHS = 40
NUM_RUNS = 5

# %% [markdown]
# # Training function

# %%
def train_single_run(run_id, seed, model_name, with_augmentation=False):
    """Train ResNet20v1 1 lần."""

    aug_label = "aug" if with_augmentation else "noaug"
    print(f"\n{'='*60}")
    print(f"  Training ResNet20v1 ({aug_label}) - Run {run_id} (seed={seed})")
    print(f"{'='*60}\n")

    if WANDB_OK:
        wandb.init(
            project="loss-landscape",
            name=f"{model_name}_run_{run_id}",
            config={
                "model": "ResNet20v1",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "run_id": run_id,
                "augmentation": with_augmentation,
            },
        )

    tf.random.set_seed(seed)
    np.random.seed(seed)

    save_dir = os.path.join(WORKING_DIR, f"{model_name}_weights", f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    tf.keras.backend.clear_session()
    model = get_training_model()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr_schedule(epoch), verbose=True
    )

    def save_model_callback(epoch, logs):
        filepath = os.path.join(save_dir, f'resnet20v1_checkpoint_{epoch}.h5')
        model.save(filepath)

    save_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=save_model_callback
    )

    callbacks = [lr_callback, save_callback]
    if WANDB_OK:
        callbacks.append(wandb.keras.WandbCallback())

    train_ds = make_dataset(x_train, y_train, shuffle=True,
                            with_augmentation=with_augmentation)
    test_ds = make_dataset(x_test, y_test, shuffle=False)

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    end = time.time()

    training_time = end - start
    print(f"\nTraining time: {training_time:.2f} seconds")

    model.save(os.path.join(save_dir, 'resnet20v1_final.h5'))

    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'training_time': training_time,
        'seed': seed,
        'run_id': run_id,
        'with_augmentation': with_augmentation,
        'nb_model_params': model.count_params(),
    }

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

    if WANDB_OK:
        wandb.finish()

    return history_dict

# %% [markdown]
# # Train KHÔNG data augmentation

# %%
seeds = [42, 123, 456, 789, 1024]
histories_noaug = []

for i, seed in enumerate(seeds):
    h = train_single_run(
        run_id=i+1, seed=seed,
        model_name='resnet20_noaug',
        with_augmentation=False
    )
    histories_noaug.append(h)

# %% [markdown]
# # Train CÓ data augmentation

# %%
histories_aug = []

for i, seed in enumerate(seeds):
    h = train_single_run(
        run_id=i+1, seed=seed,
        model_name='resnet20_aug',
        with_augmentation=True
    )
    histories_aug.append(h)

# %% [markdown]
# # Visualize kết quả

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# No augmentation
for i, h in enumerate(histories_noaug):
    axes[0, 0].plot(h['val_accuracy'], label=f'Run {i+1}', alpha=0.8)
axes[0, 0].set_title('ResNet20v1 (No Aug) - Val Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

for i, h in enumerate(histories_noaug):
    axes[0, 1].plot(h['val_loss'], label=f'Run {i+1}', alpha=0.8)
axes[0, 1].set_title('ResNet20v1 (No Aug) - Val Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# With augmentation
for i, h in enumerate(histories_aug):
    axes[1, 0].plot(h['val_accuracy'], label=f'Run {i+1}', alpha=0.8)
axes[1, 0].set_title('ResNet20v1 (Aug) - Val Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

for i, h in enumerate(histories_aug):
    axes[1, 1].plot(h['val_loss'], label=f'Run {i+1}', alpha=0.8)
axes[1, 1].set_title('ResNet20v1 (Aug) - Val Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, "resnet20_training_curves.png"), dpi=150)
plt.show()

# %%
print("\n" + "="*60)
print("  TRAINING SUMMARY - ResNet20v1")
print("="*60)

print("\n--- No Augmentation ---")
for i, h in enumerate(histories_noaug):
    print(f"Run {i+1}: val_acc={h['val_accuracy'][-1]:.4f}, time={h['training_time']:.2f}s")

print("\n--- With Augmentation ---")
for i, h in enumerate(histories_aug):
    print(f"Run {i+1}: val_acc={h['val_accuracy'][-1]:.4f}, time={h['training_time']:.2f}s")

print("\n" + "="*60)
print("All weights saved. Next: Ensemble analysis notebooks.")
