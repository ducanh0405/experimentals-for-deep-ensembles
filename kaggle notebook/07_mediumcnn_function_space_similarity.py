"""
Notebook 7: Visualizing Function Space Similarity - MediumCNN
==============================================================
Paper: "Deep Ensembles: A Loss Landscape Perspective" (Fort et al., 2019)
Platform: Kaggle (GPU P100/T4)

Mục tiêu:
- Cosine Similarity Analysis (Snapshots vs Trajectories)
- Prediction Disagreement Analysis (Snapshots vs Trajectories)
- tSNE Visualization
- Tương tự Notebook 6, áp dụng cho MediumCNN

Yêu cầu:
- Chạy sau Notebook 1 (MediumCNN training)
"""

# %%
import os
import sys

if os.path.isdir("/kaggle/working") and "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

import tensorflow as tf
import numpy as np
import json

from kaggle_utils import get_weights_dir, load_keras_model, output_path

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

print(f"TensorFlow version: {tf.__version__}")

# %% [markdown]
# # Tải CIFAR-10

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = y_test.flatten()
x_test_norm = x_test.astype(np.float32) / 255.0
print(f"Test set: {x_test_norm.shape}, {y_test.shape}")

# %% [markdown]
# # Load ALL MediumCNN checkpoints

# %%
WEIGHTS_DIR = get_weights_dir("mediumcnn_weights")
NUM_RUNS = 5
EPOCHS = 40

all_models = {}
for run_id in range(1, NUM_RUNS + 1):
    all_models[run_id] = {}
    run_dir = os.path.join(WEIGHTS_DIR, f'run_{run_id}')
    for epoch in range(EPOCHS):
        ckpt_path = os.path.join(run_dir, f'mediumcnn_checkpoint_{epoch}.h5')
        if os.path.exists(ckpt_path):
            all_models[run_id][epoch] = ckpt_path
    print(f"Run {run_id}: found {len(all_models[run_id])} checkpoints")

# %% [markdown]
# # Helper Functions

# %%
def flatten_weights(model):
    weights = model.get_weights()
    return np.concatenate([w.flatten() for w in weights])

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = norm(v1) * norm(v2)
    if norms == 0:
        return 0.0
    return dot_product / norms

def prediction_disagreement(preds1, preds2):
    return np.sum(preds1 != preds2) / len(preds1) * 100

# %% [markdown]
# # Part 1: Cosine Similarity Analysis

# %%
print("=== Cosine Similarity: SNAPSHOTS (Run 1) ===")
analysis_run = 1
snapshot_epochs = list(range(0, EPOCHS, 5))

snapshot_weights = {}
for epoch in tqdm(snapshot_epochs, desc="Loading snapshots"):
    if epoch in all_models[analysis_run]:
        model = load_keras_model(all_models[analysis_run][epoch])
        snapshot_weights[epoch] = flatten_weights(model)
        tf.keras.backend.clear_session()

n_snapshots = len(snapshot_epochs)
cos_sim_snapshots = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_weights and e2 in snapshot_weights:
            cos_sim_snapshots[i, j] = cosine_similarity(snapshot_weights[e1], snapshot_weights[e2])

print(f"  Mean off-diagonal: {np.mean(cos_sim_snapshots[np.triu_indices(n_snapshots, k=1)]):.4f}")

# %%
print("\n=== Cosine Similarity: TRAJECTORIES (Final epoch) ===")
final_epoch = EPOCHS - 1

trajectory_weights = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="Loading trajectories"):
    if final_epoch in all_models[run_id]:
        model = load_keras_model(all_models[run_id][final_epoch])
        trajectory_weights[run_id] = flatten_weights(model)
        tf.keras.backend.clear_session()

cos_sim_trajectories = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_weights and r2 in trajectory_weights:
            cos_sim_trajectories[i, j] = cosine_similarity(trajectory_weights[r1], trajectory_weights[r2])

print(f"  Mean off-diagonal: {np.mean(cos_sim_trajectories[np.triu_indices(NUM_RUNS, k=1)]):.4f}")

# %% [markdown]
# ## Cosine Similarity Heatmaps

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

im1 = axes[0].imshow(cos_sim_snapshots, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[0].set_title(f"Cosine Similarity: Snapshots (Run {analysis_run})", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Epoch")
axes[0].set_xticks(range(n_snapshots)); axes[0].set_xticklabels(snapshot_epochs)
axes[0].set_yticks(range(n_snapshots)); axes[0].set_yticklabels(snapshot_epochs)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(cos_sim_trajectories, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[1].set_title("Cosine Similarity: Trajectories (Final epoch)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Run ID"); axes[1].set_ylabel("Run ID")
axes[1].set_xticks(range(NUM_RUNS)); axes[1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[1].set_yticks(range(NUM_RUNS)); axes[1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(output_path("mediumcnn_cosine_similarity.png"), dpi=150)
plt.show()

# %% [markdown]
# # Part 2: Prediction Disagreement Analysis

# %%
print("=== Prediction Disagreement: SNAPSHOTS ===")
snapshot_preds = {}
for epoch in tqdm(snapshot_epochs, desc="Predictions snapshots"):
    if epoch in all_models[analysis_run]:
        model = load_keras_model(all_models[analysis_run][epoch])
        snapshot_preds[epoch] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_snapshots = np.zeros((n_snapshots, n_snapshots))
for i, e1 in enumerate(snapshot_epochs):
    for j, e2 in enumerate(snapshot_epochs):
        if e1 in snapshot_preds and e2 in snapshot_preds:
            disagree_snapshots[i, j] = prediction_disagreement(snapshot_preds[e1], snapshot_preds[e2])

print(f"  Mean off-diagonal: {np.mean(disagree_snapshots[np.triu_indices(n_snapshots, k=1)]):.2f}%")

# %%
print("\n=== Prediction Disagreement: TRAJECTORIES ===")
trajectory_preds = {}
for run_id in tqdm(range(1, NUM_RUNS + 1), desc="Predictions trajectories"):
    if final_epoch in all_models[run_id]:
        model = load_keras_model(all_models[run_id][final_epoch])
        trajectory_preds[run_id] = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        tf.keras.backend.clear_session()

disagree_trajectories = np.zeros((NUM_RUNS, NUM_RUNS))
for i in range(NUM_RUNS):
    for j in range(NUM_RUNS):
        r1, r2 = i + 1, j + 1
        if r1 in trajectory_preds and r2 in trajectory_preds:
            disagree_trajectories[i, j] = prediction_disagreement(trajectory_preds[r1], trajectory_preds[r2])

print(f"  Mean off-diagonal: {np.mean(disagree_trajectories[np.triu_indices(NUM_RUNS, k=1)]):.2f}%")

# %% [markdown]
# ## Prediction Disagreement Heatmaps

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

im1 = axes[0].imshow(disagree_snapshots, cmap='YlOrRd', vmin=0, vmax=30)
axes[0].set_title(f"Prediction Disagreement (%): Snapshots (Run {analysis_run})", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Epoch")
axes[0].set_xticks(range(n_snapshots)); axes[0].set_xticklabels(snapshot_epochs)
axes[0].set_yticks(range(n_snapshots)); axes[0].set_yticklabels(snapshot_epochs)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(disagree_trajectories, cmap='YlOrRd', vmin=0, vmax=30)
axes[1].set_title("Prediction Disagreement (%): Trajectories", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Run ID"); axes[1].set_ylabel("Run ID")
axes[1].set_xticks(range(NUM_RUNS)); axes[1].set_xticklabels(range(1, NUM_RUNS + 1))
axes[1].set_yticks(range(NUM_RUNS)); axes[1].set_yticklabels(range(1, NUM_RUNS + 1))
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(output_path("mediumcnn_prediction_disagreement.png"), dpi=150)
plt.show()

# %% [markdown]
# # Part 3: tSNE Visualization

# %%
print("Preparing for tSNE...")
tsne_weights = []
tsne_labels = []
tsne_colors = []
tsne_epochs = list(range(0, EPOCHS, 2))

for run_id in tqdm(range(1, NUM_RUNS + 1), desc="Loading for tSNE"):
    for epoch in tsne_epochs:
        if epoch in all_models[run_id]:
            model = load_keras_model(all_models[run_id][epoch])
            tsne_weights.append(flatten_weights(model))
            tsne_labels.append((run_id, epoch))
            tsne_colors.append(run_id)
            tf.keras.backend.clear_session()

tsne_weights = np.array(tsne_weights)
tsne_colors = np.array(tsne_colors)
print(f"tSNE input shape: {tsne_weights.shape}")

# %%
print("Running tSNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_results = tsne.fit_transform(tsne_weights)
print("tSNE complete!")

# %%
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4']
markers = ['o', 's', '^', 'D', 'v']

for run_id in range(1, NUM_RUNS + 1):
    mask = tsne_colors == run_id
    run_points = tsne_results[mask]
    ax.plot(run_points[:, 0], run_points[:, 1], color=colors[run_id-1], alpha=0.3, linewidth=1)
    ax.scatter(run_points[:, 0], run_points[:, 1], c=colors[run_id-1], marker=markers[run_id-1],
               s=40, alpha=0.7, label=f'Run {run_id}', edgecolors='white', linewidths=0.5)
    ax.scatter(run_points[0, 0], run_points[0, 1], c=colors[run_id-1], marker=markers[run_id-1],
               s=150, edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(run_points[-1, 0], run_points[-1, 1], c=colors[run_id-1], marker='*',
               s=200, edgecolors='black', linewidths=1, zorder=5)

ax.set_title("MediumCNN: tSNE of Weight Vectors\n(large dot=epoch 0, star=final epoch)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("tSNE Dimension 1", fontsize=12)
ax.set_ylabel("tSNE Dimension 2", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(output_path("mediumcnn_tsne.png"), dpi=150)
plt.show()

# %%
with open(output_path("mediumcnn_function_space_summary.json"), "w") as f:
    json.dump(
        {
            "model": "MediumCNN",
            "experiment": "function_space_similarity",
            "snapshot_cos_sim_mean": float(
                np.mean(cos_sim_snapshots[np.triu_indices(n_snapshots, k=1)])
            ),
            "trajectory_cos_sim_mean": float(
                np.mean(cos_sim_trajectories[np.triu_indices(NUM_RUNS, k=1)])
            ),
            "snapshot_disagree_mean_pct": float(
                np.mean(disagree_snapshots[np.triu_indices(n_snapshots, k=1)])
            ),
            "trajectory_disagree_mean_pct": float(
                np.mean(disagree_trajectories[np.triu_indices(NUM_RUNS, k=1)])
            ),
        },
        f,
        indent=2,
    )

snap_cos = float(np.mean(cos_sim_snapshots[np.triu_indices(n_snapshots, k=1)]))
traj_cos = float(np.mean(cos_sim_trajectories[np.triu_indices(NUM_RUNS, k=1)]))
snap_dis = float(np.mean(disagree_snapshots[np.triu_indices(n_snapshots, k=1)]))
traj_dis = float(np.mean(disagree_trajectories[np.triu_indices(NUM_RUNS, k=1)]))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].bar(["Snapshots", "Trajectories"], [snap_cos, traj_cos], color=["#4C72B0", "#DD8452"], edgecolor="black")
axes[0].set_ylabel("Mean cosine similarity")
axes[0].set_title("MediumCNN: cosine similarity (summary)")
axes[0].grid(True, axis="y", alpha=0.3)
axes[1].bar(["Snapshots", "Trajectories"], [snap_dis, traj_dis], color=["#4C72B0", "#DD8452"], edgecolor="black")
axes[1].set_ylabel("Mean prediction disagreement (%)")
axes[1].set_title("MediumCNN: prediction disagreement (summary)")
axes[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(output_path("mediumcnn_function_space_summary_bars.png"), dpi=150)
plt.show()

# %%
print("\n" + "="*60)
print("  MediumCNN FUNCTION SPACE SIMILARITY SUMMARY")
print("="*60)
print(f"\n  Cosine Similarity:")
print(f"    Snapshots mean:     {np.mean(cos_sim_snapshots[np.triu_indices(n_snapshots, k=1)]):.4f}")
print(f"    Trajectories mean:  {np.mean(cos_sim_trajectories[np.triu_indices(NUM_RUNS, k=1)]):.4f}")
print(f"\n  Prediction Disagreement:")
print(f"    Snapshots mean:     {np.mean(disagree_snapshots[np.triu_indices(n_snapshots, k=1)]):.2f}%")
print(f"    Trajectories mean:  {np.mean(disagree_trajectories[np.triu_indices(NUM_RUNS, k=1)]):.2f}%")
print("\n" + "="*60)
print("\nNext: ResNet20v1 Function Space Similarity (Notebook 8)")
