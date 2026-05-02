"""
Notebook 3: SmallCNN Val Acc Ensembles
=======================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Load tất cả SmallCNN model weights đã train (5 runs)
- Tính ensemble accuracy với kích thước ensemble khác nhau (1, 2, 3, 4, 5)
- Vẽ biểu đồ Val Accuracy vs Ensemble Size
- Log kết quả lên W&B

Yêu cầu:
- Chạy sau Notebook 0 (00_setup_and_smallcnn_training.py)
- Cần add output của Notebook 0 làm Kaggle Dataset hoặc dùng chung working dir
"""

# %%
import tensorflow as tf
import numpy as np
import os
import json
import wandb

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Đăng nhập W&B
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)

print(f"TensorFlow version: {tf.__version__}")

# %% [markdown]
# # Tải CIFAR-10

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = y_test.flatten()

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")

# Normalize
x_test_norm = x_test.astype(np.float32) / 255.0

print(f"Test set: {x_test_norm.shape}, {y_test.shape}")

# %% [markdown]
# # Load SmallCNN models (5 runs)

# %%
# Đường dẫn tới weights
# Nếu dùng Kaggle Dataset (output từ Notebook 0):
# WEIGHTS_DIR = '/kaggle/input/smallcnn-weights/smallcnn_weights'
# Nếu dùng chung working directory:
WEIGHTS_DIR = '/kaggle/working/smallcnn_weights'

NUM_RUNS = 5
EPOCHS = 40

# Load model cuối cùng của mỗi run
members = []
for run_id in range(1, NUM_RUNS + 1):
    model_path = os.path.join(WEIGHTS_DIR, f'run_{run_id}', 'smallcnn_final.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        members.append(model)
        print(f"Loaded run_{run_id}: {model_path}")
    else:
        print(f"WARNING: Model not found: {model_path}")

print(f"\nTotal models loaded: {len(members)}")

# %% [markdown]
# # Evaluate Single Model

# %%
# Chọn 1 model ngẫu nhiên để so sánh
single_model_idx = np.random.choice(len(members))
single_model = members[single_model_idx]

# Evaluate single model
single_preds = np.argmax(single_model.predict(x_test_norm), axis=1)
accuracy_single_model = 100 * accuracy_score(y_test, single_preds)
print(f"Single model (run {single_model_idx+1}) accuracy: {accuracy_single_model:.2f}%")

# %% [markdown]
# # Ensemble Predictions

# %%
def ensemble_predictions(members, x_data):
    """Tính ensemble predictions bằng probability averaging."""
    yhats = [model.predict(x_data) for model in members]
    yhats = np.array(yhats)
    # Average probabilities across ensemble members
    averaged = np.mean(yhats, axis=0)
    # Argmax để lấy class predictions
    result = np.argmax(averaged, axis=1)
    return result

def evaluate_n_members(members, x_data, y_true):
    """Evaluate ensemble accuracy cho từng ensemble size."""
    accuracy_list = []
    for n_members in range(1, len(members) + 1):
        subset = members[:n_members]
        yhat = ensemble_predictions(subset, x_data)
        acc = 100 * accuracy_score(y_true, yhat)
        accuracy_list.append(acc)
        print(f"  Ensemble size {n_members}: accuracy = {acc:.2f}%")
    return accuracy_list

# %%
print("Evaluating SmallCNN ensembles...")
accuracy_list = evaluate_n_members(members, x_test_norm, y_test)

# %% [markdown]
# # Visualize Results

# %%
ensemble_sizes = list(range(1, len(members) + 1))

plt.figure(figsize=(10, 7))
plt.plot(ensemble_sizes, accuracy_list, 'bo-', linewidth=2, markersize=8,
         label='Deep Ensemble')
plt.axhline(y=accuracy_single_model, color='r', linestyle='--', linewidth=2,
            label=f'Single Model ({accuracy_single_model:.2f}%)')

plt.title("SmallCNN: Test Accuracy as a Function of Ensemble Size",
          fontsize=14, fontweight='bold')
plt.xlabel("Ensemble Size", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.xticks(ensemble_sizes)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('/kaggle/working/smallcnn_ensemble_accuracy.png', dpi=150)
plt.show()

# %%
# Log lên W&B
wandb.init(
    project="loss-landscape",
    name="smallcnn_ensemble_accuracy",
    config={
        "model": "SmallCNN",
        "num_members": len(members),
        "experiment": "ensemble_accuracy"
    }
)

# Log accuracy cho từng ensemble size
for size, acc in zip(ensemble_sizes, accuracy_list):
    wandb.log({"ensemble_size": size, "ensemble_accuracy": acc})

# Log hình ảnh
wandb.log({"ensemble_accuracy_plot": wandb.Image('/kaggle/working/smallcnn_ensemble_accuracy.png')})

wandb.finish()

# %%
# Tổng kết
print("\n" + "="*60)
print("  SmallCNN ENSEMBLE ACCURACY SUMMARY")
print("="*60)
print(f"\n  Single model accuracy: {accuracy_single_model:.2f}%")
for size, acc in zip(ensemble_sizes, accuracy_list):
    improvement = acc - accuracy_single_model
    print(f"  Ensemble {size} models:   {acc:.2f}% (Δ = {improvement:+.2f}%)")
print("\n" + "="*60)
print("\nNext: Run MediumCNN Ensemble Accuracy (Notebook 4)")
