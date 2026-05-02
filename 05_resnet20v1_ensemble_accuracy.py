"""
Notebook 5: ResNet20v1 Val Acc Ensembles
=========================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Load tất cả ResNet20v1 model weights (cả No Aug và Aug versions)
- Tính ensemble accuracy cho cả 2 phiên bản
- So sánh ensemble accuracy giữa No Aug và Aug
- Log kết quả lên W&B

Yêu cầu:
- Chạy sau Notebook 2 (02_resnet20v1_training.py)
"""

# %%
import tensorflow as tf
import numpy as np
import os
import json
import subprocess
import wandb

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Đăng nhập W&B
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)

print(f"TensorFlow version: {tf.__version__}")

# %% [markdown]
# # Download ResNet module & Load CIFAR-10

# %%
# Download resnet_cifar10 module (cần cho custom objects khi load model)
if not os.path.exists('resnet_cifar10.py'):
    subprocess.run([
        'wget',
        'https://raw.githubusercontent.com/GoogleCloudPlatform/keras-idiomatic-programmer/master/zoo/resnet/resnet_cifar10.py'
    ])

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
            model = tf.keras.models.load_model(model_path)
            members.append(model)
            print(f"  Loaded: {model_path}")
        else:
            print(f"  WARNING: Not found: {model_path}")
    return members

# Load No Augmentation models
print("\n--- Loading ResNet20v1 (No Augmentation) ---")
members_noaug = load_models('/kaggle/working/resnet20_noaug_weights', NUM_RUNS)
print(f"Loaded {len(members_noaug)} models")

# Load With Augmentation models
print("\n--- Loading ResNet20v1 (With Augmentation) ---")
members_aug = load_models('/kaggle/working/resnet20_aug_weights', NUM_RUNS)
print(f"Loaded {len(members_aug)} models")

# %% [markdown]
# # Ensemble Analysis Functions

# %%
def ensemble_predictions(members, x_data):
    """Tính ensemble predictions bằng probability averaging."""
    yhats = [model.predict(x_data) for model in members]
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
accuracy_noaug = evaluate_n_members(members_noaug, x_test_norm, y_test, "No Aug")

print("\n=== Evaluating ResNet20v1 (With Augmentation) ===")
accuracy_aug = evaluate_n_members(members_aug, x_test_norm, y_test, "Aug")

# %% [markdown]
# # Visualize Results

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# No Augmentation
sizes_noaug = list(range(1, len(members_noaug) + 1))
axes[0].plot(sizes_noaug, accuracy_noaug, 'bo-', linewidth=2, markersize=8,
             label='Deep Ensemble')
axes[0].axhline(y=accuracy_noaug[0], color='r', linestyle='--', linewidth=2,
                label=f'Single Model ({accuracy_noaug[0]:.2f}%)')
axes[0].set_title("ResNet20v1 (No Augmentation)\nTest Accuracy vs Ensemble Size",
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel("Ensemble Size", fontsize=12)
axes[0].set_ylabel("Test Accuracy (%)", fontsize=12)
axes[0].set_xticks(sizes_noaug)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# With Augmentation
sizes_aug = list(range(1, len(members_aug) + 1))
axes[1].plot(sizes_aug, accuracy_aug, 'go-', linewidth=2, markersize=8,
             label='Deep Ensemble')
axes[1].axhline(y=accuracy_aug[0], color='r', linestyle='--', linewidth=2,
                label=f'Single Model ({accuracy_aug[0]:.2f}%)')
axes[1].set_title("ResNet20v1 (With Augmentation)\nTest Accuracy vs Ensemble Size",
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel("Ensemble Size", fontsize=12)
axes[1].set_ylabel("Test Accuracy (%)", fontsize=12)
axes[1].set_xticks(sizes_aug)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/resnet20v1_ensemble_accuracy.png', dpi=150)
plt.show()

# %% [markdown]
# # Combined Comparison Plot

# %%
plt.figure(figsize=(10, 7))

sizes = list(range(1, min(len(members_noaug), len(members_aug)) + 1))
plt.plot(sizes, accuracy_noaug[:len(sizes)], 'bo-', linewidth=2, markersize=8,
         label='No Augmentation')
plt.plot(sizes, accuracy_aug[:len(sizes)], 'go-', linewidth=2, markersize=8,
         label='With Augmentation')

plt.title("ResNet20v1: Ensemble Accuracy Comparison\nNo Aug vs With Aug",
          fontsize=14, fontweight='bold')
plt.xlabel("Ensemble Size", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.xticks(sizes)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('/kaggle/working/resnet20v1_ensemble_comparison.png', dpi=150)
plt.show()

# %%
# Log lên W&B
wandb.init(
    project="loss-landscape",
    name="resnet20v1_ensemble_accuracy",
    config={
        "model": "ResNet20v1",
        "experiment": "ensemble_accuracy"
    }
)

for size, acc_na, acc_a in zip(sizes, accuracy_noaug[:len(sizes)], accuracy_aug[:len(sizes)]):
    wandb.log({
        "ensemble_size": size,
        "noaug_accuracy": acc_na,
        "aug_accuracy": acc_a
    })

wandb.log({
    "ensemble_accuracy_plot": wandb.Image('/kaggle/working/resnet20v1_ensemble_accuracy.png'),
    "ensemble_comparison_plot": wandb.Image('/kaggle/working/resnet20v1_ensemble_comparison.png')
})
wandb.finish()

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
