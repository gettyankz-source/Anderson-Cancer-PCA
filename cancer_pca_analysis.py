"""
Anderson Cancer Center — PCA & Logistic Regression Analysis
============================================================
Assignment: Principal Component Analysis on the Breast Cancer Dataset
Modules:    sklearn, pandas, numpy, matplotlib, seaborn

Tasks Covered
─────────────
1. PCA Implementation    — identify essential variables from 30-feature dataset
2. Dimensionality Reduction — reduce to 2 PCA components
3. Bonus: Logistic Regression — classification prediction with evaluation
"""

# ─────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Shared visual style
# ─────────────────────────────────────────────────────────────
ACCENT      = "#0077B6"      # Anderson blue
ACCENT2     = "#E63946"      # malignant red
BG          = "#0D1117"
PANEL       = "#161B22"
TEXT        = "#E6EDF3"
GRID        = "#21262D"
PALETTE     = [ACCENT, ACCENT2]

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : PANEL,
    "axes.labelcolor"  : TEXT,
    "xtick.color"      : TEXT,
    "ytick.color"      : TEXT,
    "text.color"       : TEXT,
    "grid.color"       : GRID,
    "grid.linewidth"   : 0.5,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

print("=" * 65)
print("  ANDERSON CANCER CENTER — PCA ANALYSIS")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# TASK 1 — PCA IMPLEMENTATION
#   Load the cancer dataset, standardise features,
#   run full PCA to identify essential variables.
# ═════════════════════════════════════════════════════════════
print("\n[TASK 1]  PCA IMPLEMENTATION")
print("-" * 65)

# 1a. Load dataset
cancer      = load_breast_cancer()
X           = cancer.data
y           = cancer.target
features    = cancer.feature_names
target_names= cancer.target_names          # ['malignant', 'benign']

df_raw = pd.DataFrame(X, columns=features)
df_raw["target"] = y
df_raw["diagnosis"] = df_raw["target"].map({0: "Malignant", 1: "Benign"})

print(f"\n  Dataset        : Breast Cancer Wisconsin (sklearn)")
print(f"  Samples        : {X.shape[0]}")
print(f"  Features       : {X.shape[1]}")
print(f"  Classes        : {list(target_names)}")
print(f"  Class balance  : Malignant={sum(y==0)}, Benign={sum(y==1)}")

print("\n  --- Statistical Summary (first 5 features) ---")
print(df_raw[list(features[:5])].describe().round(4).to_string())

# 1b. Standardise (PCA is scale-sensitive)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n  Features standardised (mean=0, std=1) via StandardScaler.")

# 1c. Full PCA (all 30 components) to find explained variance
pca_full = PCA(n_components=30, random_state=42)
pca_full.fit(X_scaled)

evr        = pca_full.explained_variance_ratio_
cumulative = np.cumsum(evr)

print("\n  --- Explained Variance by Component ---")
for i, (v, c) in enumerate(zip(evr, cumulative), 1):
    bar = "#" * int(v * 50)
    print(f"  PC{i:>2}: {v*100:5.2f}%  (cumulative {c*100:6.2f}%)  {bar}")

n_95 = int(np.argmax(cumulative >= 0.95)) + 1
n_99 = int(np.argmax(cumulative >= 0.99)) + 1
print(f"\n  Components needed for 95% variance : {n_95}")
print(f"  Components needed for 99% variance : {n_99}")

# ── Chart 1: Scree Plot (explained variance) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA — Explained Variance Analysis",
             fontsize=15, fontweight="bold", color=ACCENT, y=1.01)

# Individual variance
ax = axes[0]
bars = ax.bar(range(1, 31), evr * 100, color=ACCENT, alpha=0.8, edgecolor="none")
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Explained Variance (%)", fontsize=11)
ax.set_title("Individual Explained Variance", fontsize=12, color=TEXT)
ax.set_xticks(range(1, 31, 2))

# Cumulative variance
ax2 = axes[1]
ax2.plot(range(1, 31), cumulative * 100, marker="o", color=ACCENT,
         linewidth=2.2, markersize=5)
ax2.axhline(95, color=ACCENT2, linestyle="--", linewidth=1.3, label="95% threshold")
ax2.axhline(99, color="#F4A261", linestyle="--", linewidth=1.3, label="99% threshold")
ax2.axvline(2, color="#57CC99", linestyle=":", linewidth=1.5, label="2 components (Task 2)")
ax2.set_xlabel("Number of Components", fontsize=11)
ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
ax2.set_title("Cumulative Explained Variance", fontsize=12, color=TEXT)
ax2.legend(fontsize=9, framealpha=0.3)
ax2.set_xticks(range(1, 31, 2))

fig.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "01_scree_plot.png")
fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\n  Chart saved -> {p1}")

# ── Chart 2: Feature loadings heatmap (PC1 & PC2) ────────────
loadings = pd.DataFrame(
    pca_full.components_[:2].T,
    index=features,
    columns=["PC1", "PC2"],
)
print("\n  --- Top 10 Features by PC1 Loading (absolute) ---")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10).round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 11))
sns.heatmap(
    loadings, annot=True, fmt=".3f", cmap="coolwarm",
    center=0, linewidths=0.4, linecolor=GRID,
    ax=ax, cbar_kws={"shrink": 0.6},
    annot_kws={"size": 8}
)
ax.set_title("Feature Loadings — PC1 & PC2",
             fontsize=14, fontweight="bold", color=ACCENT, pad=14)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "02_feature_loadings.png")
fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Chart saved -> {p2}")

# ═════════════════════════════════════════════════════════════
# TASK 2 — DIMENSIONALITY REDUCTION TO 2 PCA COMPONENTS
# ═════════════════════════════════════════════════════════════
print("\n\n[TASK 2]  DIMENSIONALITY REDUCTION  (30 features -> 2 PCA components)")
print("-" * 65)

pca2          = PCA(n_components=2, random_state=42)
X_pca2        = pca2.fit_transform(X_scaled)

var_pc1 = pca2.explained_variance_ratio_[0] * 100
var_pc2 = pca2.explained_variance_ratio_[1] * 100
var_tot = var_pc1 + var_pc2

print(f"\n  PC1 explained variance : {var_pc1:.2f}%")
print(f"  PC2 explained variance : {var_pc2:.2f}%")
print(f"  Combined (2 PCs)       : {var_tot:.2f}%")
print(f"\n  Original shape : {X.shape}")
print(f"  Reduced shape  : {X_pca2.shape}")

df_pca = pd.DataFrame(X_pca2, columns=["PC1", "PC2"])
df_pca["target"]    = y
df_pca["diagnosis"] = df_pca["target"].map({0: "Malignant", 1: "Benign"})

print("\n  --- 2-Component PCA DataFrame (first 5 rows) ---")
print(df_pca.head().to_string(index=False))

# ── Chart 3: 2D PCA Scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for label, color in zip(["Benign", "Malignant"], [ACCENT, ACCENT2]):
    subset = df_pca[df_pca["diagnosis"] == label]
    ax.scatter(
        subset["PC1"], subset["PC2"],
        label=label, color=color,
        alpha=0.65, s=40, edgecolors="none",
    )

ax.set_xlabel(f"PC1  ({var_pc1:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2  ({var_pc2:.1f}% variance)", fontsize=12)
ax.set_title(
    f"Breast Cancer Dataset — 2-Component PCA\n"
    f"Total variance explained: {var_tot:.1f}%",
    fontsize=14, fontweight="bold", color=ACCENT, pad=12,
)
ax.legend(fontsize=11, framealpha=0.3)
ax.axhline(0, color=GRID, linewidth=0.8)
ax.axvline(0, color=GRID, linewidth=0.8)
fig.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "03_pca2_scatter.png")
fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\n  Chart saved -> {p3}")

# ── Chart 4: PC1 & PC2 distribution by class ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PC1 & PC2 Distribution by Diagnosis",
             fontsize=14, fontweight="bold", color=ACCENT)

for ax, comp in zip(axes, ["PC1", "PC2"]):
    for label, color in zip(["Benign", "Malignant"], [ACCENT, ACCENT2]):
        data = df_pca[df_pca["diagnosis"] == label][comp]
        ax.hist(data, bins=30, alpha=0.55, color=color, label=label, edgecolor="none")
    ax.set_xlabel(comp, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{comp} Distribution", fontsize=12, color=TEXT)
    ax.legend(fontsize=10, framealpha=0.3)

fig.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "04_component_distributions.png")
fig.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Chart saved -> {p4}")

# ═════════════════════════════════════════════════════════════
# BONUS — LOGISTIC REGRESSION FOR PREDICTION
#   Train on the 2 PCA components, evaluate fully.
# ═════════════════════════════════════════════════════════════
print("\n\n[BONUS]   LOGISTIC REGRESSION — PREDICTION")
print("-" * 65)

# Split BEFORE PCA to avoid data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale and project using training-fit only
scaler_lr    = StandardScaler()
X_train_sc   = scaler_lr.fit_transform(X_train_raw)
X_test_sc    = scaler_lr.transform(X_test_raw)

pca_lr       = PCA(n_components=2, random_state=42)
X_train_pca  = pca_lr.fit_transform(X_train_sc)
X_test_pca   = pca_lr.transform(X_test_sc)

# Train logistic regression on PCA features
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_pca, y_train)

y_pred      = lr_model.predict(X_test_pca)
y_prob      = lr_model.predict_proba(X_test_pca)[:, 1]
accuracy    = accuracy_score(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)
report      = classification_report(y_test, y_pred, target_names=target_names)

print(f"\n  Train set size : {X_train_pca.shape[0]} samples")
print(f"  Test  set size : {X_test_pca.shape[0]} samples")
print(f"\n  Test Accuracy  : {accuracy * 100:.2f}%")
print(f"\n  Classification Report:\n{report}")
print(f"  Confusion Matrix:\n{cm}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
print(f"\n  ROC-AUC Score  : {roc_auc:.4f}")

# ── Chart 5: Decision boundary ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

h = 0.05
x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.18, cmap="coolwarm")
ax.contour(xx, yy, Z, colors=[TEXT], linewidths=0.8, alpha=0.4)

for label, color, marker in zip(
    [0, 1], [ACCENT2, ACCENT], ["x", "o"]
):
    mask = y_train == label
    ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
               color=color, alpha=0.55, s=35,
               marker=marker, label=f"{target_names[label]} (train)",
               edgecolors="none")

ax.set_xlabel(f"PC1  ({pca_lr.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2  ({pca_lr.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("Logistic Regression — Decision Boundary (PCA Space)",
             fontsize=14, fontweight="bold", color=ACCENT, pad=12)
ax.legend(fontsize=10, framealpha=0.3)
fig.tight_layout()
p5 = os.path.join(OUTPUT_DIR, "05_decision_boundary.png")
fig.savefig(p5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\n  Chart saved -> {p5}")

# ── Chart 6: Confusion matrix ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, linewidths=0.5, linecolor=GRID,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
ax.set_ylabel("True Label", fontsize=12, labelpad=8)
ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy*100:.2f}%)",
             fontsize=13, fontweight="bold", color=ACCENT, pad=12)
fig.tight_layout()
p6 = os.path.join(OUTPUT_DIR, "06_confusion_matrix.png")
fig.savefig(p6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Chart saved -> {p6}")

# ── Chart 7: ROC Curve ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"ROC Curve  (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], color=GRID, linestyle="--", lw=1.2, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Logistic Regression on 2 PCA Components",
             fontsize=13, fontweight="bold", color=ACCENT, pad=12)
ax.legend(fontsize=11, framealpha=0.3)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
fig.tight_layout()
p7 = os.path.join(OUTPUT_DIR, "07_roc_curve.png")
fig.savefig(p7, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Chart saved -> {p7}")

# ── Chart 8: Test predictions scatter ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Logistic Regression — Test Set Results (PCA Space)",
             fontsize=14, fontweight="bold", color=ACCENT)

titles = ["True Labels", "Predicted Labels"]
label_sets = [y_test, y_pred]

for ax, lbls, title in zip(axes, label_sets, titles):
    for cls, color in zip([0, 1], [ACCENT2, ACCENT]):
        mask = lbls == cls
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   color=color, alpha=0.7, s=50, edgecolors="none",
                   label=target_names[cls])
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=12, color=TEXT)
    ax.legend(fontsize=10, framealpha=0.3)
    ax.axhline(0, color=GRID, lw=0.7)
    ax.axvline(0, color=GRID, lw=0.7)

fig.tight_layout()
p8 = os.path.join(OUTPUT_DIR, "08_test_predictions.png")
fig.savefig(p8, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Chart saved -> {p8}")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ALL TASKS COMPLETE")
print(f"  Accuracy: {accuracy*100:.2f}%   |   ROC-AUC: {roc_auc:.4f}")
print(f"  Outputs -> {OUTPUT_DIR}")
print("=" * 65)
