"""
Notebook 1: MediumCNN Training
================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Train MediumCNN trên CIFAR-10 với 5 random initializations
- Lưu model weights mỗi epoch
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

from kaggle_utils import WORKING_DIR, tf_autotune, wandb_login_optional

WANDB_OK = wandb_login_optional()
wandb = None
if WANDB_OK:

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

    try:
        import wandb as _wandb
    except ImportError:
        install("wandb")
        import wandb as _wandb
    wandb = _wandb

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# # Tải và chuẩn bị CIFAR-10

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")

AUTO = tf_autotune()
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

# %% [markdown]
# # Định nghĩa MediumCNN

# %%
def MediumCNN():
    """
    MediumCNN architecture theo paper:
    - 4 Conv2D layers (32 → 64 → 128 → 128 filters)
    - Dropout(0.1) và MaxPooling2D sau mỗi conv layer
    - GlobalAveragePooling2D
    - Dense(32) + Dropout(0.1)
    - Dense(10, softmax)
    Total params: ~89,226
    """
    inputs = keras.layers.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3))

    x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = keras.activations.relu(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(64, (3, 3), padding='valid')(x)
    x = keras.activations.relu(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = keras.activations.relu(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = keras.activations.relu(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(10, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

model = MediumCNN()
model.summary()

# %% [markdown]
# # Cấu hình Training

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
BASE_SAVE_DIR = os.path.join(WORKING_DIR, "mediumcnn_weights")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# %% [markdown]
# # Training Loop

# %%
def train_single_run(run_id, seed):
    """Train MediumCNN 1 lần với seed cho trước."""

    print(f"\n{'='*60}")
    print(f"  Training MediumCNN - Run {run_id} (seed={seed})")
    print(f"{'='*60}\n")

    if WANDB_OK:
        wandb.init(
            project="loss-landscape",
            name=f"mediumcnn_run_{run_id}",
            config={
                "model": "MediumCNN",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "run_id": run_id,
            },
        )

    tf.random.set_seed(seed)
    np.random.seed(seed)

    save_dir = os.path.join(BASE_SAVE_DIR, f'run_{run_id}')
    os.makedirs(save_dir, exist_ok=True)

    keras.backend.clear_session()
    model = MediumCNN()
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr_schedule(epoch), verbose=True
    )

    def save_model_callback(epoch, logs):
        filepath = os.path.join(save_dir, f'mediumcnn_checkpoint_{epoch}.h5')
        model.save(filepath)

    save_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=save_model_callback
    )

    callbacks = [lr_callback, save_callback]
    if WANDB_OK:
        callbacks.append(wandb.keras.WandbCallback())

    train_ds = make_dataset(x_train, y_train, shuffle=True)
    test_ds = make_dataset(x_test, y_test, shuffle=False)

    start = time.time()
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    end = time.time()

    training_time = end - start
    print(f"\nTraining time: {training_time:.2f} seconds")

    model.save(os.path.join(save_dir, 'mediumcnn_final.h5'))

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

    if WANDB_OK:
        wandb.finish()

    return history_dict

# %%
all_histories = []
seeds = [42, 123, 456, 789, 1024]

for i, seed in enumerate(seeds):
    history = train_single_run(run_id=i+1, seed=seed)
    all_histories.append(history)

# %% [markdown]
# # Visualize kết quả

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, h in enumerate(all_histories):
    axes[0].plot(h['val_accuracy'], label=f'Run {i+1} (seed={seeds[i]})', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title('MediumCNN - Validation Accuracy across runs')
axes[0].legend()
axes[0].grid(True)

for i, h in enumerate(all_histories):
    axes[1].plot(h['val_loss'], label=f'Run {i+1} (seed={seeds[i]})', alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('MediumCNN - Validation Loss across runs')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()

# %%
print("\n" + "="*60)
print("  TRAINING SUMMARY - MediumCNN")
print("="*60)
for i, h in enumerate(all_histories):
    print(f"\nRun {i+1} (seed={seeds[i]}):")
    print(f"  Final val_accuracy: {h['val_accuracy'][-1]:.4f}")
    print(f"  Final val_loss:     {h['val_loss'][-1]:.4f}")
    print(f"  Training time:      {h['training_time']:.2f}s")
print("\n" + "="*60)
print(f"\nWeights saved at: {BASE_SAVE_DIR}")
print("Next: Run ResNet20v1 training (Notebook 2)")
