"""
Notebook 8: Visualizing Function Space Similarity - ResNet20v1
===============================================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Cosine Similarity Analysis (Snapshots vs Trajectories) cho cả No Aug và Aug
- Prediction Disagreement Analysis (Snapshots vs Trajectories)
- LƯU Ý tSNE: Vì vector weight của ResNet20 khá lớn, tSNE có thể gặp lỗi MemoryError trên Kaggle (13GB RAM).
  Chúng ta sẽ sử dụng UMAP (nhanh hơn, ít bộ nhớ hơn) hoặc downsample weight vector nếu cần.

Yêu cầu:
- Chạy sau Notebook 2 (ResNet20v1 training)
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
    ensure_umap,
    get_weights_dir,
    load_keras_model,
    output_path,
)

from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

print(f"TensorFlow version: {tf.__version__}")

ensure_umap()
import umap

# %% [markdown]
# # Download ResNet module & Load CIFAR-10

# %%
ensure_resnet_cifar10_module()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_test = y_test.flatten()
x_test_norm = tf.image.convert_image_dtype(x_test, tf.float32).numpy()
print(f"Test set: {x_test_norm.shape}, {y_test.shape}")

# %% [markdown]
# # Helper Functions

# %%
def flatten_weights(model, subsample=1.0):
    """
    Flatten weights. Nếu model quá lớn, có thể subsample (random sample 1 phần weight)
    để tránh MemoryError khi chạy tSNE/UMAP.
    Với ResNet20 (~270k params), thường RAM 13GB vẫn xử lý được, nhưng ta để option.
    """
    weights = model.get_weights()
    flat = np.concatenate([w.flatten() for w in weights])
    if subsample < 1.0:
        np.random.seed(42)
        idx = np.random.choice(len(flat), size=int(len(flat)*subsample), replace=False)
        return flat[idx]
    return flat

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = norm(v1) * norm(v2)
    if norms == 0: return 0.0
    return dot_product / norms

def prediction_disagreement(preds1, preds2):
    return np.sum(preds1 != preds2) / len(preds1) * 100

def get_checkpoints_dict(weights_dir, num_runs, epochs):
    models_dict = {}
    for run_id in range(1, num_runs + 1):
        models_dict[run_id] = {}
        run_dir = os.path.join(weights_dir, f'run_{run_id}')
        for epoch in range(epochs):
            ckpt_path = os.path.join(run_dir, f'resnet20v1_checkpoint_{epoch}.h5')
            if os.path.exists(ckpt_path):
                models_dict[run_id][epoch] = ckpt_path
    return models_dict

# %% [markdown]
# # Phân tích: ResNet20v1 (NO Augmentation)

# %%
NUM_RUNS = 5
EPOCHS = 40
analysis_run = 1

print("\n" + "="*50)
print("  PART A: NO AUGMENTATION")
print("="*50)

weights_noaug = get_weights_dir('resnet20_noaug_weights')
all_models_noaug = get_checkpoints_dict(weights_noaug, NUM_RUNS, EPOCHS)

# 1. Cosine Similarity - Snapshots
snapshot_epochs = list(range(0, EPOCHS, 5))
snapshot_weights_noaug = {}
for epoch in tqdm(snapshot_epochs, desc="[NoAug] Loading snapshots"):
    if epoch in all_models_noaug[analysis_run]:
        model = load_keras_model(all_models_noaug[analysis_run][epoch])
        snapshot_weights_noaug[epoch] = flatten_weights(model)
        tf.keras.backend.clear_session()

n_snapshots = len(snapshot_epochs)
cos_sim_snap_noaug = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_weights_noaug and e2 in snapshot_weights_noaug:
            cos_sim_snap_noaug[i, j] = cosine_similarity(snapshot_weights_noaug[e1], snapshot_weights_noaug[e2])

# 2. Cosine Similarity - Trajectories
final_epoch = EPOCHS - 1
trajectory_weights_noaug = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="[NoAug] Loading trajectories"):
    if final_epoch in all_models_noaug[run_id]:
        model = load_keras_model(all_models_noaug[run_id][final_epoch])
        trajectory_weights_noaug[run_id] = flatten_weights(model)
        tf.keras.backend.clear_session()

cos_sim_traj_noaug = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_weights_noaug and r2 in trajectory_weights_noaug:
            cos_sim_traj_noaug[i, j] = cosine_similarity(trajectory_weights_noaug[r1], trajectory_weights_noaug[r2])

# 3. Prediction Disagreement - Snapshots
snapshot_preds_noaug = {}
for epoch in tqdm(snapshot_epochs, desc="[NoAug] Preds snapshots"):
    if epoch in all_models_noaug[analysis_run]:
        model = load_keras_model(all_models_noaug[analysis_run][epoch])
        snapshot_preds_noaug[epoch] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_snap_noaug = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_preds_noaug and e2 in snapshot_preds_noaug:
            disagree_snap_noaug[i, j] = prediction_disagreement(snapshot_preds_noaug[e1], snapshot_preds_noaug[e2])

# 4. Prediction Disagreement - Trajectories
trajectory_preds_noaug = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="[NoAug] Preds trajectories"):
    if final_epoch in all_models_noaug[run_id]:
        model = load_keras_model(all_models_noaug[run_id][final_epoch])
        trajectory_preds_noaug[run_id] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_traj_noaug = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_preds_noaug and r2 in trajectory_preds_noaug:
            disagree_traj_noaug[i, j] = prediction_disagreement(trajectory_preds_noaug[r1], trajectory_preds_noaug[r2])


# %% [markdown]
# # Phân tích: ResNet20v1 (WITH Augmentation)

# %%
print("\n" + "="*50)
print("  PART B: WITH AUGMENTATION")
print("="*50)

weights_aug = get_weights_dir('resnet20_aug_weights')
all_models_aug = get_checkpoints_dict(weights_aug, NUM_RUNS, EPOCHS)

# 1. Cosine Similarity - Snapshots
snapshot_weights_aug = {}
for epoch in tqdm(snapshot_epochs, desc="[Aug] Loading snapshots"):
    if epoch in all_models_aug[analysis_run]:
        model = load_keras_model(all_models_aug[analysis_run][epoch])
        snapshot_weights_aug[epoch] = flatten_weights(model)
        tf.keras.backend.clear_session()

cos_sim_snap_aug = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_weights_aug and e2 in snapshot_weights_aug:
            cos_sim_snap_aug[i, j] = cosine_similarity(snapshot_weights_aug[e1], snapshot_weights_aug[e2])

# 2. Cosine Similarity - Trajectories
trajectory_weights_aug = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="[Aug] Loading trajectories"):
    if final_epoch in all_models_aug[run_id]:
        model = load_keras_model(all_models_aug[run_id][final_epoch])
        trajectory_weights_aug[run_id] = flatten_weights(model)
        tf.keras.backend.clear_session()

cos_sim_traj_aug = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_weights_aug and r2 in trajectory_weights_aug:
            cos_sim_traj_aug[i, j] = cosine_similarity(trajectory_weights_aug[r1], trajectory_weights_aug[r2])

# 3. Prediction Disagreement - Snapshots
snapshot_preds_aug = {}
for epoch in tqdm(snapshot_epochs, desc="[Aug] Preds snapshots"):
    if epoch in all_models_aug[analysis_run]:
        model = load_keras_model(all_models_aug[analysis_run][epoch])
        snapshot_preds_aug[epoch] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_snap_aug = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_preds_aug and e2 in snapshot_preds_aug:
            disagree_snap_aug[i, j] = prediction_disagreement(snapshot_preds_aug[e1], snapshot_preds_aug[e2])

# 4. Prediction Disagreement - Trajectories
trajectory_preds_aug = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="[Aug] Preds trajectories"):
    if final_epoch in all_models_aug[run_id]:
        model = load_keras_model(all_models_aug[run_id][final_epoch])
        trajectory_preds_aug[run_id] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_traj_aug = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_preds_aug and r2 in trajectory_preds_aug:
            disagree_traj_aug[i, j] = prediction_disagreement(trajectory_preds_aug[r1], trajectory_preds_aug[r2])

# %% [markdown]
# # Visualize: Cosine Similarity

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# No Aug
im1 = axes[0, 0].imshow(cos_sim_snap_noaug, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[0, 0].set_title(f"[No Aug] Cosine Similarity: Snapshots", fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(n_snapshots)); axes[0, 0].set_xticklabels(snapshot_epochs)
axes[0, 0].set_yticks(range(n_snapshots)); axes[0, 0].set_yticklabels(snapshot_epochs)
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(cos_sim_traj_noaug, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[0, 1].set_title("[No Aug] Cosine Similarity: Trajectories", fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(NUM_RUNS)); axes[0, 1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[0, 1].set_yticks(range(NUM_RUNS)); axes[0, 1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im2, ax=axes[0, 1])

# Aug
im3 = axes[1, 0].imshow(cos_sim_snap_aug, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[1, 0].set_title(f"[Aug] Cosine Similarity: Snapshots", fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(n_snapshots)); axes[1, 0].set_xticklabels(snapshot_epochs)
axes[1, 0].set_yticks(range(n_snapshots)); axes[1, 0].set_yticklabels(snapshot_epochs)
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(cos_sim_traj_aug, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[1, 1].set_title("[Aug] Cosine Similarity: Trajectories", fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(NUM_RUNS)); axes[1, 1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[1, 1].set_yticks(range(NUM_RUNS)); axes[1, 1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(output_path("resnet20_cosine_similarity.png"), dpi=150)
plt.show()

# %% [markdown]
# # Visualize: Prediction Disagreement

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# No Aug
im1 = axes[0, 0].imshow(disagree_snap_noaug, cmap='YlOrRd', vmin=0, vmax=30)
axes[0, 0].set_title(f"[No Aug] Prediction Disagreement (%): Snapshots", fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(n_snapshots)); axes[0, 0].set_xticklabels(snapshot_epochs)
axes[0, 0].set_yticks(range(n_snapshots)); axes[0, 0].set_yticklabels(snapshot_epochs)
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(disagree_traj_noaug, cmap='YlOrRd', vmin=0, vmax=30)
axes[0, 1].set_title("[No Aug] Prediction Disagreement (%): Trajectories", fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(NUM_RUNS)); axes[0, 1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[0, 1].set_yticks(range(NUM_RUNS)); axes[0, 1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im2, ax=axes[0, 1])

# Aug
im3 = axes[1, 0].imshow(disagree_snap_aug, cmap='YlOrRd', vmin=0, vmax=30)
axes[1, 0].set_title(f"[Aug] Prediction Disagreement (%): Snapshots", fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(n_snapshots)); axes[1, 0].set_xticklabels(snapshot_epochs)
axes[1, 0].set_yticks(range(n_snapshots)); axes[1, 0].set_yticklabels(snapshot_epochs)
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(disagree_traj_aug, cmap='YlOrRd', vmin=0, vmax=30)
axes[1, 1].set_title("[Aug] Prediction Disagreement (%): Trajectories", fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(NUM_RUNS)); axes[1, 1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[1, 1].set_yticks(range(NUM_RUNS)); axes[1, 1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(output_path("resnet20_prediction_disagreement.png"), dpi=150)
plt.show()

# %% [markdown]
# # UMAP / tSNE Visualization

# %%
print("\n" + "="*50)
print("  PART C: DIMENSIONALITY REDUCTION (UMAP)")
print("="*50)

# Gom weight data (chỉ cho No Augmentation để demo, do memory limits)
# Lấy subset epochs: mỗi 4 epochs
tsne_epochs = list(range(0, EPOCHS, 4))
tsne_weights = []
tsne_colors = []
tsne_labels = []

print("Preparing weights for UMAP (No Augmentation)...")
for run_id in tqdm(range(1, NUM_RUNS + 1)):
    for epoch in tsne_epochs:
        if epoch in all_models_noaug[run_id]:
            model = load_keras_model(all_models_noaug[run_id][epoch])
            # Subsample 20% weights để giảm RAM nếu cần (tuỳ chọn)
            tsne_weights.append(flatten_weights(model, subsample=0.2))
            tsne_colors.append(run_id)
            tsne_labels.append((run_id, epoch))
            tf.keras.backend.clear_session()

tsne_weights = np.array(tsne_weights)
tsne_colors = np.array(tsne_colors)
print(f"UMAP input shape: {tsne_weights.shape}")

# Chạy UMAP (thay vì tSNE, vì UMAP nhanh hơn nhiều cho vector lớn)
print("Running UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(tsne_weights)
print("UMAP complete!")

# Vẽ
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4']
markers = ['o', 's', '^', 'D', 'v']

for run_id in range(1, NUM_RUNS + 1):
    mask = tsne_colors == run_id
    if sum(mask) == 0: continue
    run_points = embedding[mask]
    ax.plot(run_points[:, 0], run_points[:, 1], color=colors[run_id-1], alpha=0.3, linewidth=1)
    ax.scatter(run_points[:, 0], run_points[:, 1], c=colors[run_id-1], marker=markers[run_id-1],
               s=60, alpha=0.8, label=f'Run {run_id}', edgecolors='white', linewidths=0.5)
    
    # Mark start and end
    ax.scatter(run_points[0, 0], run_points[0, 1], c=colors[run_id-1], marker=markers[run_id-1],
               s=200, edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(run_points[-1, 0], run_points[-1, 1], c=colors[run_id-1], marker='*',
               s=250, edgecolors='black', linewidths=1, zorder=5)

ax.set_title("ResNet20v1 (No Aug): UMAP of Weight Vectors\n(large dot=epoch 0, star=final epoch)",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(output_path("resnet20_umap.png"), dpi=150)
plt.show()

# %%
with open(output_path("resnet20_function_space_summary.json"), "w") as f:
    json.dump(
        {
            "model": "ResNet20v1",
            "experiment": "function_space_similarity",
            "noaug_snapshot_cos_sim_mean": float(
                np.mean(cos_sim_snap_noaug[np.triu_indices(n_snapshots, k=1)])
            ),
            "noaug_trajectory_cos_sim_mean": float(
                np.mean(cos_sim_traj_noaug[np.triu_indices(NUM_RUNS, k=1)])
            ),
            "aug_snapshot_cos_sim_mean": float(
                np.mean(cos_sim_snap_aug[np.triu_indices(n_snapshots, k=1)])
            ),
            "aug_trajectory_cos_sim_mean": float(
                np.mean(cos_sim_traj_aug[np.triu_indices(NUM_RUNS, k=1)])
            ),
        },
        f,
        indent=2,
    )

na_s = float(np.mean(cos_sim_snap_noaug[np.triu_indices(n_snapshots, k=1)]))
na_t = float(np.mean(cos_sim_traj_noaug[np.triu_indices(NUM_RUNS, k=1)]))
ag_s = float(np.mean(cos_sim_snap_aug[np.triu_indices(n_snapshots, k=1)]))
ag_t = float(np.mean(cos_sim_traj_aug[np.triu_indices(NUM_RUNS, k=1)]))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, [na_s, na_t], width=w, label="No Aug", color="#4C72B0", edgecolor="black")
ax.bar(x + w / 2, [ag_s, ag_t], width=w, label="Aug", color="#55A868", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["Snapshots", "Trajectories"])
ax.set_ylabel("Mean cosine similarity")
ax.set_title("ResNet20v1: cosine similarity (summary)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(output_path("resnet20_function_space_summary_bars.png"), dpi=150)
plt.show()

# %%
print("\n" + "="*60)
print("  ResNet20v1 FUNCTION SPACE SIMILARITY SUMMARY")
print("="*60)
print("\nNext: Final Summary & Comparison (Notebook 9)")
