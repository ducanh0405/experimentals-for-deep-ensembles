"""
Notebook 0: Setup & SmallCNN Training
======================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Cài đặt môi trường
- Train SmallCNN trên CIFAR-10 với multiple random initializations
- Lưu model weights mỗi epoch để phân tích snapshots sau này
"""

# %% [markdown]
# # Phase 1: Setup môi trường

# %%
# Kiểm tra GPU
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# %%
# Cài đặt thư viện cần thiết
!pip install wandb --quiet

# %%
# Đăng nhập W&B với API key
import wandb
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)

# %%
import tensorflow as tf
import numpy as np
import os
import time
import json

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %%
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# %% [markdown]
# # Phase 2: Tải và chuẩn bị CIFAR-10

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")

print(f"Train: {x_train.shape}, {y_train.shape}")
print(f"Test:  {x_test.shape}, {y_test.shape}")

# %%
# Tạo data pipeline
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_SHAPE = 32

def preprocess_image(image, label):
    img = tf.cast(image, tf.float32)
    img = img / 255.0
    return img, label

def make_dataset(x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(1024)
    ds = (ds
          .map(preprocess_image, num_parallel_calls=AUTO)
          .batch(BATCH_SIZE)
          .prefetch(AUTO))
    return ds

trainloader = make_dataset(x_train, y_train, shuffle=True)
testloader = make_dataset(x_test, y_test, shuffle=False)

# %% [markdown]
# # Phase 3: Định nghĩa SmallCNN

# %%
def SmallCNN():
    """
    SmallCNN architecture theo paper:
    - 3 Conv2D layers (16 → 32 → 32 filters)
    - MaxPooling2D sau mỗi conv layer
    - GlobalAveragePooling2D
    - Dense(32) + Dropout(0.1)
    - Dense(10, softmax)
    Total params: ~15,722
    """
    inputs = keras.layers.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3))

    x = keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D(2, strides=2)(x)

    x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D(2, strides=2)(x)

    x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPooling2D(2, strides=2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(10, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

# Kiểm tra kiến trúc
model = SmallCNN()
model.summary()

# %% [markdown]
# # Phase 4: Cấu hình Training

# %%
# LR Schedule theo paper
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

# Visualize LR schedule
rng = list(range(40))
plt.figure(figsize=(10, 4))
plt.plot(rng, [lr_schedule(x) for x in rng])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Custom LR Schedule (theo paper)')
plt.grid(True)
plt.show()

# %%
EPOCHS = 40
NUM_RUNS = 5  # Số lần train với random init khác nhau

# Thư mục lưu weights (trên Kaggle, dùng /kaggle/working/)
BASE_SAVE_DIR = '/kaggle/working/smallcnn_weights'
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# %% [markdown]
# # Phase 5: Training Loop

# %%
def train_single_run(run_id, seed):
    """Train SmallCNN 1 lần với seed cho trước."""

    print(f"\n{'='*60}")
    print(f"  Training SmallCNN - Run {run_id} (seed={seed})")
    print(f"{'='*60}\n")

    # Khởi tạo WandB cho run này
    wandb.init(
        project="loss-landscape",
        name=f"smallcnn_run_{run_id}",
        config={
            "model": "SmallCNN",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": seed,
            "run_id": run_id
        }
    )

    # Set random seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Tạo thư mục lưu weights cho run này
    save_dir = os.path.join(BASE_SAVE_DIR, f'run_{run_id}')
    os.makedirs(save_dir, exist_ok=True)

    # Tạo model mới
    keras.backend.clear_session()
    model = SmallCNN()
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr_schedule(epoch), verbose=True
    )

    def save_model_callback(epoch, logs):
        filepath = os.path.join(save_dir, f'smallcnn_checkpoint_{epoch}.h5')
        model.save(filepath)

    save_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=save_model_callback
    )

    wandb_callback = wandb.keras.WandbCallback()

    # Recreate data pipeline mỗi run (để shuffle khác nhau)
    train_ds = make_dataset(x_train, y_train, shuffle=True)
    test_ds = make_dataset(x_test, y_test, shuffle=False)

    # Train
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[lr_callback, save_callback, wandb_callback],
        verbose=1
    )
    end = time.time()

    training_time = end - start
    print(f"\nTraining time: {training_time:.2f} seconds")

    # Lưu model cuối cùng
    model.save(os.path.join(save_dir, 'smallcnn_final.h5'))

    # Lưu history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'training_time': training_time,
        'seed': seed,
        'run_id': run_id,
    }

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

    wandb.finish()

    return history_dict

# %%
# Train tất cả các runs
all_histories = []
seeds = [42, 123, 456, 789, 1024]  # 5 seeds khác nhau

for i, seed in enumerate(seeds):
    history = train_single_run(run_id=i+1, seed=seed)
    all_histories.append(history)

# %% [markdown]
# # Phase 6: Visualize kết quả training

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy
for i, h in enumerate(all_histories):
    axes[0].plot(h['val_accuracy'], label=f'Run {i+1} (seed={seeds[i]})', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title('SmallCNN - Validation Accuracy across runs')
axes[0].legend()
axes[0].grid(True)

# Plot loss
for i, h in enumerate(all_histories):
    axes[1].plot(h['val_loss'], label=f'Run {i+1} (seed={seeds[i]})', alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('SmallCNN - Validation Loss across runs')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()

# %%
# Tổng kết
print("\n" + "="*60)
print("  TRAINING SUMMARY")
print("="*60)
for i, h in enumerate(all_histories):
    print(f"\nRun {i+1} (seed={seeds[i]}):")
    print(f"  Final val_accuracy: {h['val_accuracy'][-1]:.4f}")
    print(f"  Final val_loss:     {h['val_loss'][-1]:.4f}")
    print(f"  Training time:      {h['training_time']:.2f}s")
print("\n" + "="*60)

print(f"\nTất cả weights đã được lưu tại: {BASE_SAVE_DIR}")
print("Tiếp theo: Chạy notebook MediumCNN training (Notebook 1)")
