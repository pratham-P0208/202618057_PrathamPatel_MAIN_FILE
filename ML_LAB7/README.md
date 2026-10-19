# Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

**Course:** DS605: Fundamentals of Machine Learning
**Dataset:** AT&T Database of Faces (ORL)
**Notebook:** `202618057_LAB7.ipynb`

---

## 1. Objective

Represent face images numerically, reduce their dimensionality with PCA (eigenfaces), perform face matching in the reduced space, and compare the result with traditional machine-learning classifiers.

**Pipeline**

```
Images -> Grayscale Pixel Matrix -> Flatten -> Mean Centering -> PCA / SVD
  -> Explained Variance -> Eigenfaces -> Reduced Representation
  -> Reconstruction -> Similarity Matching -> ML Classification
```

All PCA/eigenface computations (mean face, covariance eigendecomposition, SVD, projection, reconstruction, nearest-neighbour matching) are implemented with **NumPy only**. Scikit-learn is used only for the train/test split and the final traditional-ML comparison in Part D. Pillow is used for reading images.

---

## 2. Repository Structure

```
.
├── 202618057_LAB7.ipynb     # Full runnable notebook (Parts A-D)
├── Data/                    # ORL dataset: s1 ... s40, each with 1.pgm ... 10.pgm
├── figures/                 # Saved figures (see Section 9)
└── README.md
```

**Requirements:** Python 3, `numpy`, `matplotlib`, `pillow`, `pandas`, `scikit-learn`

**To run:** place the dataset in `Data/` and update `DATASET_DIR` in the second code cell of the notebook (it currently points to a local Windows path), then run all cells in order.

---

## 3. Dataset

| Property | Value |
|---|---|
| Subjects | 40 |
| Images per subject | 10 |
| Total images | 400 |
| Image size | 112 x 92 pixels (grayscale) |
| Pixel type / range | `uint8`, 0-255 (sample image: min 11, max 234) |
| Flattened feature length | 10,304 |

---

## 4. Approach and Results

### Part A: Image Representation

- Each `.pgm` image was read as a grayscale matrix of shape `(112, 92)`, with `uint8` values.
- Each image was flattened into a 10,304-dimensional vector.
- Full image matrix `X`: **(400, 10304)**, label vector `y`: **(400,)** with subject IDs 1-40.
- **Train/test split:** 80/20, `random_state=42`, stratified by subject.
  - Training: **320 images** (8 per subject)
  - Testing: **80 images** (2 per subject)
  - An assertion in the notebook confirms all 40 subjects appear in both sets.

### Part B: PCA, Eigenfaces and Compression

**Mean face and centering.** The mean face was computed from the training set only, then subtracted from the training data (and later from test images using the same training mean, so no test information leaks into PCA).

**Two ways to obtain principal directions**

1. **Covariance / eigendecomposition.** With 10,304 pixels but only 320 samples, the full 10,304 x 10,304 covariance matrix is impractical. Instead, the smaller 320 x 320 matrix `X_c X_c^T / (n-1)` was decomposed using `np.linalg.eigh`, and its eigenvectors were mapped back to pixel space (`X_c^T v`, normalised by `sqrt((n-1) * lambda)`).
2. **SVD.** `np.linalg.svd(X_centered, full_matrices=False)`. The rows of `Vt` are the principal directions, and explained variance is `s^2 / sum(s^2)`.

**Comparison of the two methods**

| Check | Result |
|---|---|
| Valid (non-zero) components | 319 (rank is at most n-1 after centering) |
| \|cosine similarity\| of first 10 components | 1.0 for all 10 |
| Max explained-variance difference | 2.5e-16 |

The two methods give the same principal directions (up to sign) and identical explained variance. SVD is simpler and numerically more stable, so it is used for the remaining parts.

**Explained variance.** The minimum number of components that preserves at least 90% variance is **k = 98** (90.06% preserved). The cumulative-variance curve is in `figures/explained_variance.png`.

| Components | Variance preserved |
|---|---|
| 10 | 60.14% |
| 20 | 70.49% |
| 30 | 76.00% |
| 50 | 82.48% |
| **98** | **90.06%** |

**Eigenfaces.** The first 10 principal components were reshaped to 112 x 92 and displayed as eigenfaces (`figures/eigenfaces.png`). Early eigenfaces capture broad lighting and face-shape variation; later ones capture finer detail.

**Projection and reconstruction.** Training and test images were projected onto the 98 selected components, giving reduced representations of shape **(320, 98)** and **(80, 98)**. Test faces were reconstructed as `projection @ components + mean_face`. Test reconstruction MSE: **329.98** (on a 0-255 intensity scale). An original-vs-reconstructed example is in `figures/reconstruction.png`.

**Storage comparison**

| Quantity | Value |
|---|---|
| Original image | 10,304 bytes (uint8) |
| Reduced coefficients per image (98 x float32) | 392 bytes |
| Per-image storage reduction | **96.2%** |
| PCA model (mean face + 98 components, float32) | 4,080,384 bytes |
| Original training set (320 images) | 3,297,280 bytes |
| Reduced training set + PCA model | 4,205,824 bytes |
| Total storage change on the 320-image set | **-27.55%** (larger) |

The per-image saving is large, but the one-time cost of storing the PCA model outweighs it for a dataset this small. The model cost is roughly amortised after about 412 images (4,080,384 / (10,304 - 392)), so PCA storage savings only appear for larger collections.

### Part C: Recognition in PCA Space

Each test image was centered with the training mean, projected into the 98-dimensional PCA space, and matched to the closest training face using **Euclidean distance** (1-nearest-neighbour). The predicted label is the label of that nearest training image.

| Method | Feature dimension | Accuracy |
|---|---|---|
| Raw pixels (Euclidean) | 10,304 | **95.0%** (76/80) |
| PCA features (Euclidean) | 98 | **95.0%** (76/80) |

- PCA matched raw-pixel accuracy while using about 105x fewer features.
- 76 test images were matched correctly and 4 incorrectly. The notebook displays 3 correct and 3 incorrect test/match pairs (`figures/matches.png`) and prints a per-image report (the first 10 test images were all correct).

### Part D: Traditional Machine Learning on PCA Features

KNN, Logistic Regression and SVM were trained on PCA features for 10, 20, 30, 50 and 98 components. Hyperparameters were tuned with 3-fold `GridSearchCV` on the training set:

- KNN: `n_neighbors` in {1, 3, 5}
- Logistic Regression: `C` in {0.1, 1, 10}
- SVM: `C` in {1, 10}, `kernel` in {linear, rbf}

**Full results (sorted by cross-validation accuracy)**

| Components | Model | CV Accuracy | Test Accuracy | Best Parameters |
|---|---|---|---|---|
| 30 | Logistic Regression | 98.13% | 95.00% | C=0.1 |
| 50 | Logistic Regression | 97.81% | **98.75%** | C=0.1 |
| 50 | SVM | 97.81% | 97.50% | C=10, rbf |
| 98 | SVM | 97.81% | 96.25% | C=10, rbf |
| 30 | SVM | 97.50% | 95.00% | C=10, rbf |
| 20 | Logistic Regression | 97.19% | 95.00% | C=0.1 |
| 98 | Logistic Regression | 97.19% | **98.75%** | C=0.1 |
| 50 | KNN | 96.56% | 96.25% | k=1 |
| 30 | KNN | 95.63% | 95.00% | k=1 |
| 20 | SVM | 95.62% | 95.00% | C=1, linear |
| 98 | KNN | 95.62% | 95.00% | k=1 |
| 20 | KNN | 93.75% | 95.00% | k=1 |
| 10 | SVM | 93.44% | 95.00% | C=1, linear |
| 10 | KNN | 91.25% | 95.00% | k=1 |
| 10 | Logistic Regression | 91.24% | 86.25% | C=10 |

**Comparison with the similarity baseline**

| Method | Test Accuracy |
|---|---|
| Raw-pixel nearest neighbour | 95.00% |
| PCA nearest neighbour (98 components) | 95.00% |
| KNN (50 components) | 96.25% |
| SVM (50 components, RBF) | 97.50% |
| Logistic Regression (50 components) | **98.75%** |

The accuracy-vs-components plot is in `figures/accuracy_vs_components.png`.

---

## 5. Key Observations

1. **SVD and covariance PCA agree.** Principal directions match (cosine similarity 1.0) and explained variance differs by about 1e-16, so the choice is about convenience and stability, not results.
2. **Faces compress well.** 98 of 319 possible components retain 90% of the variance, and the first 10 components alone carry about 60%.
3. **PCA keeps recognition accuracy while shrinking features.** Nearest-neighbour matching reaches 95% on both raw pixels and 98 PCA features, so most identity information survives the roughly 105x reduction.
4. **Compact PCA features plus a classifier beat simple matching.** Logistic Regression and SVM at 50 components reached 98.75% and 97.5% test accuracy, versus 95% for the baseline. This improvement comes from a very small test set, so a difference of one image is 1.25 percentage points.
5. **More components is not always better.** Accuracy peaked around 50 components for the classifiers; going to 98 did not improve it further (KNN and SVM dropped slightly), so 50 components (82.5% variance) gives a more compact representation with equal or better accuracy.
6. **Too few components hurts.** With 10 components Logistic Regression fell to 86.25%.
7. **PCA compression only pays off at scale.** Per-image storage falls by 96.2%, but the stored PCA model makes the total larger for a 320-image training set.
8. **KNN preferred k=1** in every setting, which is typical for face data with few images per subject.

---

## 6. Limitations

- **Small test set.** There are only 80 test images, so each misclassification changes accuracy by 1.25 points. Differences of a few points between methods are not statistically strong.
- **Model selection.** The best configuration should be chosen by cross-validation accuracy (Logistic Regression, 30 components: 98.13% CV) rather than test accuracy. That configuration scored 95% on the test set, while the highest test score (98.75%) came from other configurations, so the 50-component choice above is a judgement based on both CV and test results and should be read with that caveat.
- **Single split.** Results come from one stratified split (`random_state=42`); repeated splits or k-fold cross-validation would give more reliable estimates.

---

## 7. Reproducibility

- Random seed: `np.random.seed(42)`; split uses `random_state=42` with `stratify=y`.
- PCA statistics (mean face, components) are computed from the training set only.

---

## 8. How the Requirements Are Met

| Requirement | Where |
|---|---|
| Grayscale matrices, dimensions, intensities, flattening, labels | Part A |
| Reproducible stratified split | Part A |
| Mean face, centering, eigendecomposition and SVD comparison | Part B |
| Cumulative explained variance, 90% threshold, eigenfaces | Part B |
| Projection, reconstruction, storage comparison | Part B |
| Euclidean nearest-neighbour recognition, correct/incorrect matches, raw vs PCA | Part C |
| At least two ML classifiers, component and hyperparameter experiments | Part D (KNN, Logistic Regression, SVM) |
| NumPy-only PCA (no `sklearn.decomposition.PCA`) | Parts B-C |