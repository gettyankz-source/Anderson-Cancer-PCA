# Anderson Cancer Center — PCA & Logistic Regression Analysis

> **Assignment project** — Principal Component Analysis · Dimensionality Reduction · Logistic Regression Prediction

---

## Project Structure

```
Anderson_Cancer_PCA/
│
├── cancer_pca_analysis.py    <- Main Python script (all tasks)
│
├── outputs/                  <- Generated charts (auto-created on first run)
│   ├── 01_scree_plot.png               # Task 1 — explained variance
│   ├── 02_feature_loadings.png         # Task 1 — PC1 & PC2 feature loadings
│   ├── 03_pca2_scatter.png             # Task 2 — 2-component PCA scatter
│   ├── 04_component_distributions.png  # Task 2 — PC1 & PC2 histograms
│   ├── 05_decision_boundary.png        # Bonus — logistic regression boundary
│   ├── 06_confusion_matrix.png         # Bonus — confusion matrix
│   ├── 07_roc_curve.png                # Bonus — ROC curve
│   └── 08_test_predictions.png         # Bonus — test set results
│
└── README.md
```

---

## Dataset

**Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (cell nucleus measurements) |
| Classes | Malignant (212) · Benign (357) |
| Source | UCI Machine Learning Repository |

Each sample captures 10 real-valued measurements of cell nuclei computed from digitised fine needle aspirate (FNA) images, with mean, standard error, and worst (largest) values recorded — giving 30 features total.

---

## Task Coverage

### Task 1 — PCA Implementation
- Loads the breast cancer dataset from `sklearn.datasets`
- Applies `StandardScaler` (required before PCA — features are on different scales)
- Runs full PCA across all 30 components
- Prints explained variance per component and cumulative totals
- Identifies that **10 components** capture 95% variance and **17 components** capture 99%
- Outputs the **Scree Plot** and **Feature Loadings Heatmap** (PC1 & PC2)

**Top features identified by PC1 loading:**
`mean concave points`, `mean concavity`, `worst concave points`, `mean compactness`, `worst perimeter`

### Task 2 — Dimensionality Reduction (30 → 2 components)
- Applies `PCA(n_components=2)` to reduce 30 features to 2 principal components
- PC1 explains **44.27%** of variance; PC2 explains **18.97%** (combined: **63.24%**)
- Outputs a 2D scatter plot clearly separating Malignant vs Benign clusters
- Outputs histogram distributions of PC1 and PC2 by diagnosis class

### Bonus — Logistic Regression Prediction
- Proper pipeline: **train/test split → StandardScaler → PCA → LogisticRegression**
  (fit only on training data to avoid data leakage)
- 80/20 stratified split (455 train / 114 test)
- **Test Accuracy: 94.74%**  |  **ROC-AUC: 0.9888**
- Outputs: decision boundary, confusion matrix, ROC curve, test predictions plot

---

## Results Summary

| Metric | Value |
|---|---|
| PCA components for 95% variance | 10 |
| 2-component combined variance | 63.24% |
| Logistic Regression accuracy | **94.74%** |
| ROC-AUC | **0.9888** |
| Malignant precision / recall | 0.91 / 0.95 |
| Benign precision / recall | 0.97 / 0.94 |

---

## Requirements

```
Python >= 3.9
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

```bash
# From the project folder
python cancer_pca_analysis.py
```

The script will:
1. Load and describe the breast cancer dataset
2. Standardise features and run full 30-component PCA
3. Print explained variance for every component
4. Reduce to 2 PCA components
5. Run logistic regression and print accuracy, classification report, ROC-AUC
6. Save all 8 charts to `outputs/`

No external data files needed — the dataset is loaded directly from `sklearn.datasets`.

---

## Key Findings

- **PC1 is dominated by concavity and compactness features** — the most diagnostically meaningful measurements for separating malignant from benign tumours.
- **2 components alone capture 63%** of the total variance and are sufficient to visually separate the two classes.
- Logistic regression trained on just 2 PCA components achieves **94.74% accuracy** — demonstrating that the dimensionality reduction preserves the essential discriminative information.
- The **ROC-AUC of 0.989** confirms near-perfect ranking ability, which is critical for a cancer screening context where false negatives carry high clinical cost.

---

*Submitted as a ZIP archive per assignment instructions.*
