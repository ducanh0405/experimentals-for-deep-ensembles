"""
Notebook 5: ResNet20v1 Val Acc Ensembles
=========================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Load tất cả ResNet20v1 model weights (cả No Aug và Aug versions)
- Tính ensemble accuracy cho cả 2 phiên bản
- So sánh ensemble accuracy giữa No Aug và Aug (đường + biểu đồ cột)

Yêu cầu:
- Chạy sau Notebook 2 (02_resnet20v1_training.py)
"""

# %%
import os
import sys

if os.path.isdir("/kaggle/working") and "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

import tensorflow as tf
import numpy as np
import json

from kaggle_utils import (
    ensure_resnet_cifar10_module,
    get_weights_dir,
    load_keras_model,
    output_path,
)

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

print(f"TensorFlow version: {tf.__version__}")

# %% [markdown]
# # Download ResNet module & Load CIFAR-10

# %%
ensure_resnet_cifar10_module()

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_test = y_test.flatten()

# Normalize theo cách giống training (convert_image_dtype)
x_test_norm = tf.image.convert_image_dtype(x_test, tf.float32).numpy()
print(f"Test set: {x_test_norm.shape}, {y_test.shape}")

# %% [markdown]
# # Load ResNet20v1 models

# %%
NUM_RUNS = 5

def load_models(weights_dir, num_runs):
    """Load models from a given weights directory."""
    members = []
    for run_id in range(1, num_runs + 1):
        model_path = os.path.join(weights_dir, f'run_{run_id}', 'resnet20v1_final.h5')
        if os.path.exists(model_path):
            model = load_keras_model(model_path)
            members.append(model)
            print(f"  Loaded: {model_path}")
        else:
            print(f"  WARNING: Not found: {model_path}")
    return members

# Load No Augmentation models
print("\n--- Loading ResNet20v1 (No Augmentation) ---")
weights_noaug = get_weights_dir("resnet20_noaug_weights")
members_noaug = load_models(weights_noaug, NUM_RUNS)
print(f"Loaded {len(members_noaug)} models")

# Load With Augmentation models
print("\n--- Loading ResNet20v1 (With Augmentation) ---")
weights_aug = get_weights_dir("resnet20_aug_weights")
members_aug = load_models(weights_aug, NUM_RUNS)
print(f"Loaded {len(members_aug)} models")

if len(members_noaug) == 0 and len(members_aug) == 0:
    raise FileNotFoundError(
        "Không tìm thấy ResNet weights. Chạy notebook 2 hoặc gắn resnet20_*_weights vào Input."
    )

# %% [markdown]
# # Ensemble Analysis Functions

# %%
def ensemble_predictions(members, x_data):
    """Tính ensemble predictions bằng probability averaging."""
    yhats = [model.predict(x_data, verbose=0) for model in members]
    yhats = np.array(yhats)
    averaged = np.mean(yhats, axis=0)
    result = np.argmax(averaged, axis=1)
    return result

def evaluate_n_members(members, x_data, y_true, label=""):
    """Evaluate ensemble accuracy cho từng ensemble size."""
    accuracy_list = []
    for n_members in range(1, len(members) + 1):
        subset = members[:n_members]
        yhat = ensemble_predictions(subset, x_data)
        acc = 100 * accuracy_score(y_true, yhat)
        accuracy_list.append(acc)
        print(f"  [{label}] Ensemble size {n_members}: accuracy = {acc:.2f}%")
    return accuracy_list

# %% [markdown]
# # Evaluate Ensembles

# %%
print("\n=== Evaluating ResNet20v1 (No Augmentation) ===")
accuracy_noaug = (
    evaluate_n_members(members_noaug, x_test_norm, y_test, "No Aug")
    if members_noaug
    else []
)

print("\n=== Evaluating ResNet20v1 (With Augmentation) ===")
accuracy_aug = (
    evaluate_n_members(members_aug, x_test_norm, y_test, "Aug")
    if members_aug
    else []
)

# %% [markdown]
# # Visualize Results

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# No Augmentation
sizes_noaug = list(range(1, len(members_noaug) + 1))
if accuracy_noaug:
    axes[0].plot(sizes_noaug, accuracy_noaug, 'bo-', linewidth=2, markersize=8,
                 label='Deep Ensemble')
    axes[0].axhline(y=accuracy_noaug[0], color='r', linestyle='--', linewidth=2,
                    label=f'Single Model ({accuracy_noaug[0]:.2f}%)')
else:
    axes[0].text(0.5, 0.5, "No weights", ha='center', va='center', transform=axes[0].transAxes)
axes[0].set_title("ResNet20v1 (No Augmentation)\nTest Accuracy vs Ensemble Size",
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel("Ensemble Size", fontsize=12)
axes[0].set_ylabel("Test Accuracy (%)", fontsize=12)
axes[0].set_xticks(sizes_noaug)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# With Augmentation
sizes_aug = list(range(1, len(members_aug) + 1))
if accuracy_aug:
    axes[1].plot(sizes_aug, accuracy_aug, 'go-', linewidth=2, markersize=8,
                 label='Deep Ensemble')
    axes[1].axhline(y=accuracy_aug[0], color='r', linestyle='--', linewidth=2,
                    label=f'Single Model ({accuracy_aug[0]:.2f}%)')
else:
    axes[1].text(0.5, 0.5, "No weights", ha='center', va='center', transform=axes[1].transAxes)
axes[1].set_title("ResNet20v1 (With Augmentation)\nTest Accuracy vs Ensemble Size",
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel("Ensemble Size", fontsize=12)
axes[1].set_ylabel("Test Accuracy (%)", fontsize=12)
axes[1].set_xticks(sizes_aug)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path("resnet20v1_ensemble_accuracy.png"), dpi=150)
plt.show()

# %% [markdown]
# # Combined Comparison Plot

# %%
plt.figure(figsize=(10, 7))

sizes = []
if members_noaug and members_aug:
    sizes = list(range(1, min(len(members_noaug), len(members_aug)) + 1))
elif members_noaug:
    sizes = list(range(1, len(members_noaug) + 1))
elif members_aug:
    sizes = list(range(1, len(members_aug) + 1))

if members_noaug:
    plt.plot(
        list(range(1, len(members_noaug) + 1)),
        accuracy_noaug,
        'bo-',
        linewidth=2,
        markersize=8,
        label='No Augmentation',
    )
if members_aug:
    plt.plot(
        list(range(1, len(members_aug) + 1)),
        accuracy_aug,
        'go-',
        linewidth=2,
        markersize=8,
        label='With Augmentation',
    )

plt.title("ResNet20v1: Ensemble Accuracy Comparison\nNo Aug vs With Aug",
          fontsize=14, fontweight='bold')
plt.xlabel("Ensemble Size", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
if sizes:
    plt.xticks(sizes)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

comparison_path = output_path("resnet20v1_ensemble_comparison.png")
plt.savefig(comparison_path, dpi=150)
plt.show()

if accuracy_noaug and len(accuracy_noaug) > 0:
    base_na = accuracy_noaug[0]
    deltas_na = [a - base_na for a in accuracy_noaug]
else:
    deltas_na = []
if accuracy_aug and len(accuracy_aug) > 0:
    base_a = accuracy_aug[0]
    deltas_a = [a - base_a for a in accuracy_aug]
else:
    deltas_a = []

if deltas_na or deltas_a:
    plt.figure(figsize=(9, 4.5))
    w = 0.35
    if deltas_na and deltas_a and len(deltas_na) == len(deltas_a):
        x = np.arange(1, len(deltas_na) + 1)
        plt.bar(x - w / 2, deltas_na, width=w, label="No Aug Δ vs single", color="#4C72B0", edgecolor="black")
        plt.bar(x + w / 2, deltas_a, width=w, label="Aug Δ vs single", color="#55A868", edgecolor="black")
        plt.xticks(x)
    elif deltas_na:
        x = np.arange(1, len(deltas_na) + 1)
        plt.bar(x, deltas_na, label="No Aug Δ vs single", color="#4C72B0", edgecolor="black")
        plt.xticks(x)
    else:
        x = np.arange(1, len(deltas_a) + 1)
        plt.bar(x, deltas_a, label="Aug Δ vs single", color="#55A868", edgecolor="black")
        plt.xticks(x)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Ensemble size")
    plt.ylabel("Δ accuracy (percentage points)")
    plt.title("ResNet20v1: ensemble gain vs single model (same branch)", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path("resnet20v1_ensemble_delta.png"), dpi=150)
    plt.show()

with open(output_path("resnet20v1_ensemble_summary.json"), "w") as f:
    json.dump(
        {
            "model": "ResNet20v1",
            "experiment": "ensemble_accuracy",
            "noaug": {"ensemble_sizes": sizes_noaug, "test_accuracy_pct": accuracy_noaug},
            "aug": {"ensemble_sizes": sizes_aug, "test_accuracy_pct": accuracy_aug},
        },
        f,
        indent=2,
    )

# %%
print("\n" + "="*60)
print("  ResNet20v1 ENSEMBLE ACCURACY SUMMARY")
print("="*60)

print("\n  --- No Augmentation ---")
for size, acc in zip(sizes_noaug, accuracy_noaug):
    print(f"  Ensemble {size}: {acc:.2f}%")

print("\n  --- With Augmentation ---")
for size, acc in zip(sizes_aug, accuracy_aug):
    print(f"  Ensemble {size}: {acc:.2f}%")

print("\n" + "="*60)
print("\nPhase 3 (Ensemble Accuracy) COMPLETE.")
print("Next: Phase 4 - Function Space Similarity Analysis")
