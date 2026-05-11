"""
Notebook 9: Summary & Conclusions
==================================
Đọc kết quả JSON trong WORKING_DIR (từ notebook 3–8), hiển thị bảng và vẽ biểu đồ matplotlib.
"""

# %%
import json
import os
import sys

if os.path.isdir("/kaggle/working") and "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kaggle_utils import WORKING_DIR, display_df, output_path


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ensemble_acc_data = []
func_sim_data = []


def append_local_ensemble():
    specs = [
        ("smallcnn_ensemble_summary.json", "SmallCNN", "Standard"),
        ("mediumcnn_ensemble_summary.json", "MediumCNN", "Standard"),
        ("resnet20v1_ensemble_summary.json", "ResNet20v1", None),
    ]
    for fname, model_name, typ in specs:
        d = _load_json(os.path.join(WORKING_DIR, fname))
        if not d:
            continue
        if model_name == "ResNet20v1" and isinstance(d.get("noaug"), dict):
            for split, key in (("No Aug", "noaug"), ("Aug", "aug")):
                block = d.get(key) or {}
                for sz, acc in zip(block.get("ensemble_sizes") or [], block.get("test_accuracy_pct") or []):
                    ensemble_acc_data.append({
                        "Model": model_name,
                        "Type": split,
                        "Ensemble Size": sz,
                        "Accuracy": acc,
                    })
        else:
            for sz, acc in zip(d.get("ensemble_sizes") or [], d.get("test_accuracy_pct") or []):
                ensemble_acc_data.append({
                    "Model": model_name,
                    "Type": typ or "Standard",
                    "Ensemble Size": sz,
                    "Accuracy": acc,
                })


def append_local_function_space():
    for fname, model, mode in [
        ("smallcnn_function_space_summary.json", "SmallCNN", "std"),
        ("mediumcnn_function_space_summary.json", "MediumCNN", "std"),
        ("resnet20_function_space_summary.json", "ResNet20v1", "res"),
    ]:
        d = _load_json(os.path.join(WORKING_DIR, fname))
        if not d:
            continue
        if mode == "std":
            func_sim_data.append({
                "Model": model,
                "Type": "Standard",
                "Snapshots CosSim": d.get("snapshot_cos_sim_mean", np.nan),
                "Trajectories CosSim": d.get("trajectory_cos_sim_mean", np.nan),
                "Snapshots Disagree (%)": d.get("snapshot_disagree_mean_pct", np.nan),
                "Trajectories Disagree (%)": d.get("trajectory_disagree_mean_pct", np.nan),
            })
        else:
            func_sim_data.append({
                "Model": model,
                "Type": "No Aug",
                "Snapshots CosSim": d.get("noaug_snapshot_cos_sim_mean", np.nan),
                "Trajectories CosSim": d.get("noaug_trajectory_cos_sim_mean", np.nan),
            })
            func_sim_data.append({
                "Model": model,
                "Type": "Aug",
                "Snapshots CosSim": d.get("aug_snapshot_cos_sim_mean", np.nan),
                "Trajectories CosSim": d.get("aug_trajectory_cos_sim_mean", np.nan),
            })


append_local_ensemble()
append_local_function_space()

df_acc = pd.DataFrame(ensemble_acc_data)
df_sim = pd.DataFrame(func_sim_data)
if not df_acc.empty:
    df_acc = df_acc.drop_duplicates()
if not df_sim.empty:
    df_sim = df_sim.drop_duplicates()


def plot_ensemble_curves(df):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, typ), grp in df.groupby(["Model", "Type"]):
        g = grp.sort_values("Ensemble Size")
        label = f"{model} ({typ})"
        ax.plot(g["Ensemble Size"], g["Accuracy"], marker="o", linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Ensemble size", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("Summary: ensemble test accuracy", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path("summary_ensemble_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved chart: {out}")


def plot_function_space_bars(df):
    if df.empty:
        return
    has_dis = (
        df["Snapshots Disagree (%)"].notna() & df["Trajectories Disagree (%)"].notna()
    ).any()
    nrows = 2 if has_dis else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), squeeze=False)
    ax0 = axes[0, 0]
    labels = [f"{r['Model']}\n({r['Type']})" for _, r in df.iterrows()]
    x = np.arange(len(df))
    w = 0.35
    snap = df["Snapshots CosSim"].astype(float).values
    traj = df["Trajectories CosSim"].astype(float).values
    ax0.bar(x - w / 2, snap, width=w, label="Snapshots cos sim", color="#4C72B0", edgecolor="black")
    ax0.bar(x + w / 2, traj, width=w, label="Trajectories cos sim", color="#DD8452", edgecolor="black")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("Cosine similarity")
    ax0.set_title("Summary: mean cosine similarity", fontsize=13, fontweight="bold")
    ax0.legend()
    ax0.grid(True, axis="y", alpha=0.3)
    if has_dis:
        ax1 = axes[1, 0]
        d1 = df["Snapshots Disagree (%)"].astype(float).values
        d2 = df["Trajectories Disagree (%)"].astype(float).values
        ax1.bar(x - w / 2, d1, width=w, label="Snapshots disagree", color="#4C72B0", edgecolor="black")
        ax1.bar(x + w / 2, d2, width=w, label="Trajectories disagree", color="#DD8452", edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel("Disagreement (%)")
        ax1.set_title("Summary: mean prediction disagreement", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_path("summary_function_space_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved chart: {out}")


# %% [markdown]
# # 1. Ensemble Accuracy Summary

# %%
if not df_acc.empty:
    print("\n--- ENSEMBLE ACCURACY ---")
    pivot_acc = df_acc.pivot_table(
        index=["Model", "Type"],
        columns="Ensemble Size",
        values="Accuracy",
    ).round(2)
    display_df(pivot_acc)
    cols = pivot_acc.columns.tolist()
    c1 = 1 if 1 in cols else (1.0 if 1.0 in cols else None)
    c5 = 5 if 5 in cols else (5.0 if 5.0 in cols else None)
    if c1 is not None and c5 is not None:
        pivot_boost = pivot_acc.copy()
        pivot_boost["Boost (1->5)"] = pivot_boost[c5] - pivot_boost[c1]
        print("\nAccuracy boost (ensemble 1 → 5):")
        display_df(pivot_boost[["Boost (1->5)"]])
    plot_ensemble_curves(df_acc)
else:
    print(
        "No ensemble accuracy data. Run notebooks 3–5 first; JSON files should appear in WORKING_DIR."
    )

# %% [markdown]
# # 2. Function Space Similarity Summary

# %%
if not df_sim.empty:
    print("\n--- FUNCTION SPACE SIMILARITY ---")
    display_df(df_sim.round(4))
    print("\nVerification checks (expectations from paper):")
    for _, row in df_sim.iterrows():
        model_name = f"{row['Model']} ({row['Type']})"
        if pd.notna(row.get("Snapshots CosSim")) and pd.notna(row.get("Trajectories CosSim")):
            cos_check = "PASS" if row["Snapshots CosSim"] > row["Trajectories CosSim"] else "FAIL"
            print(f"  {model_name}: Snapshots cos sim > Trajectories cos sim? -> {cos_check}")
        if pd.notna(row.get("Snapshots Disagree (%)")) and pd.notna(row.get("Trajectories Disagree (%)")):
            dis_check = "PASS" if row["Snapshots Disagree (%)"] < row["Trajectories Disagree (%)"] else "FAIL"
            print(f"  {model_name}: Snapshots disagree < Trajectories disagree? -> {dis_check}")
    plot_function_space_bars(df_sim)
else:
    print("No function space data. Run notebooks 6–8 first.")

# %% [markdown]
# # 3. Kết luận (định tính)
#
# Xem paper và báo cáo `github code/README.md` của repo LossLandscape.
