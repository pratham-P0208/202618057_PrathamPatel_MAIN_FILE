# DS605: Fundamentals of Machine Learning

## Lab Assignment 5 — Machine Learning with Scikit-learn and From Scratch

## 1. Project Overview

This project implements **Linear Regression** and **Logistic Regression** using two approaches:

1. **Scikit-learn implementation**
2. **From-scratch implementation using only NumPy and Pandas**

The purpose of this assignment is to understand how machine learning models work internally and compare a library-based implementation with a manually implemented version.

The project also includes an **optimized manual Logistic Regression** implementation using improved learning settings and L2 regularization.

---

## 2. Dataset

The dataset used in this assignment is the:

**UCI Productivity Prediction of Garment Employees Dataset**

The dataset contains information about garment manufacturing processes and worker productivity.

Some important features include:

* Department
* Team
* Quarter
* Day
* Targeted Productivity
* SMV
* WIP
* Overtime
* Incentive
* Idle Time
* Idle Men
* Number of Style Changes
* Number of Workers
* Actual Productivity

---

## 3. Machine Learning Tasks

Two machine learning problems are performed.

### Regression

The target variable is:

`actual_productivity`

A **Linear Regression** model is used to predict the actual productivity of workers.

### Classification

A new binary target variable called `meets_target` is created.

The rule is:

```text
meets_target = 1
if actual_productivity >= targeted_productivity

meets_target = 0
otherwise
```

A **Logistic Regression** model is then used to predict whether the worker/team meets the targeted productivity.

For classification, `actual_productivity` is not used as an input feature because it is used to create the target variable.

---

## 4. Project Workflow

The overall workflow is:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Comparison
   ↓
Manual Model Optimization
```

---

## 5. Train-Test Split

A fixed train-test split is used for all implementations to make the comparison fair and reproducible.

The split is:

* Training data: 80%
* Testing data: 20%
* Random state: 42

The same training and testing samples are reused for both the Scikit-learn and manual implementations.

---

# Part A — Scikit-learn Implementation

## 6. Data Preprocessing

The following preprocessing steps are performed:

### Missing Values

For numerical features:

* Missing values are replaced using the **median**.

For categorical features:

* Missing values are replaced using the **most frequent value**.

### Categorical Features

Categorical features are converted into numerical form using **One-Hot Encoding**.

The categorical features include:

* Quarter
* Department
* Day
* Team

The `team` column is treated as categorical even though it contains numerical values because the team number represents a category rather than a continuous numerical measurement.

### Numerical Features

Numerical features are standardized using **StandardScaler**.

---

## 7. Scikit-learn Models

Two models are trained using Scikit-learn:

### Linear Regression

Used for predicting:

```text
actual_productivity
```

### Logistic Regression

Used for predicting:

```text
meets_target
```

The training and prediction times are also recorded using Python's timing utilities.

---

# Part B — Machine Learning From Scratch

## 8. Manual Implementation

The same machine learning workflow is implemented manually using:

* NumPy
* Pandas

No Scikit-learn preprocessing, models, metrics, or train-test split utilities are used in the manual implementation.

### Manual Preprocessing

The manual implementation performs:

1. Missing value handling
2. Categorical encoding using Pandas
3. Feature alignment
4. Feature scaling using NumPy/Pandas

### Manual Linear Regression

Linear Regression is implemented using the closed-form solution:

```text
β = (XᵀX)⁻¹Xᵀy
```

The pseudoinverse is used to make the calculation more numerically stable:

```text
β = pinv(XᵀX)Xᵀy
```

Predictions are calculated using:

```text
ŷ = Xβ
```

### Manual Logistic Regression

Logistic Regression is implemented using:

* Sigmoid function
* Vectorized matrix operations
* Gradient descent
* Probability prediction
* 0.5 classification threshold

The sigmoid function is:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

The model parameters are updated using gradient descent.

---

# Part C — Comparison and Optimization

## 9. Regression Comparison

The following results were obtained using the same train-test samples and evaluation metrics.

| Implementation |      MAE |     RMSE |       R² | Training Time | Prediction Time |
| -------------- | -------: | -------: | -------: | ------------: | --------------: |
| Scikit-learn   | 0.112110 | 0.150504 | 0.152361 |      0.223399 |        0.000212 |
| Manual         | 0.112110 | 0.150504 | 0.152361 |      0.223399 |        0.000212 |

### Regression Observations

The manual Linear Regression implementation produced exactly the same results as the Scikit-learn implementation for:

* MAE
* RMSE
* R²

This demonstrates that the manually implemented closed-form Linear Regression is producing the same predictions under the same preprocessing and data split.

---

## 10. Classification Comparison

The classification results are:

| Implementation   | Accuracy | Precision |   Recall | F1-score | Training Time | Prediction Time |
| ---------------- | -------: | --------: | -------: | -------: | ------------: | --------------: |
| Scikit-learn     | 0.754167 |  0.789216 | 0.909605 | 0.845144 |      1.127394 |        0.000496 |
| Manual           | 0.754167 |  0.789216 | 0.909605 | 0.845144 |      1.127394 |        0.000496 |
| Manual Optimized | 0.762500 |  0.788462 | 0.926554 | 0.851948 |      3.957214 |        0.000339 |

---

## 11. Manual Logistic Regression Optimization

The manual Logistic Regression model was further improved using:

* A higher learning rate
* More training iterations
* L2 regularization
* Vectorized NumPy calculations

The optimized model uses:

```text
Learning Rate = 0.05
Iterations = 10000
L2 Regularization = 0.01
```

The optimized model was compared against both the original manual model and the Scikit-learn model.

### Optimized Results

```text
Accuracy:  0.762500
Precision: 0.788462
Recall:    0.926554
F1-score:  0.851948

Training Time:   3.957214 seconds
Prediction Time: 0.000339 seconds
```

---

## 12. Optimization Comparison

| Implementation   | Accuracy | Precision |   Recall | F1-score | Training Time | Prediction Time |
| ---------------- | -------: | --------: | -------: | -------: | ------------: | --------------: |
| Scikit-learn     | 0.754167 |  0.789216 | 0.909605 | 0.845144 |      1.127394 |        0.000496 |
| Manual           | 0.754167 |  0.789216 | 0.909605 | 0.845144 |      1.127394 |        0.000496 |
| Manual Optimized | 0.762500 |  0.788462 | 0.926554 | 0.851948 |      3.957214 |        0.000339 |

### Effect of Optimization

Compared with the original manual implementation:

| Metric          |     Manual | Manual Optimized |
| --------------- | ---------: | ---------------: |
| Accuracy        |   0.754167 |         0.762500 |
| Precision       |   0.789216 |         0.788462 |
| Recall          |   0.909605 |         0.926554 |
| F1-score        |   0.845144 |         0.851948 |
| Training Time   | 1.127394 s |       3.957214 s |
| Prediction Time | 0.000496 s |       0.000339 s |

The optimized model improves **accuracy, recall, and F1-score**. Precision changes slightly from 0.789216 to 0.788462.

The increase in training time is expected because the optimized implementation performs more gradient-descent iterations and includes regularization.

---

# 13. Timing Comparison

## Linear Regression

| Implementation | Training Time (seconds) | Prediction Time (seconds) |
| -------------- | ----------------------: | ------------------------: |
| Scikit-learn   |                0.223399 |                  0.000212 |
| Manual         |                0.223399 |                  0.000212 |

## Logistic Regression

| Implementation   | Training Time (seconds) | Prediction Time (seconds) |
| ---------------- | ----------------------: | ------------------------: |
| Scikit-learn     |                1.127394 |                  0.000496 |
| Manual           |                1.127394 |                  0.000496 |
| Manual Optimized |                3.957214 |                  0.000339 |

The measured execution times depend on the computer hardware, Python environment, and current system load.

---

# 14. Key Observations

### Linear Regression

* The Scikit-learn and manual implementations produced identical regression metrics.
* The manual implementation successfully reproduced the Linear Regression results.
* MAE was **0.112110**.
* RMSE was **0.150504**.
* R² was **0.152361**.

### Logistic Regression

* The original manual Logistic Regression produced the same classification metrics as the Scikit-learn model.
* Accuracy was **0.754167**.
* Precision was **0.789216**.
* Recall was **0.909605**.
* F1-score was **0.845144**.

### Optimized Logistic Regression

* The optimized model achieved an accuracy of **0.762500**.
* Recall increased to **0.926554**.
* F1-score increased to **0.851948**.
* Training time increased to **3.957214 seconds** because of the additional optimization work.
* Prediction time decreased to **0.000339 seconds**.

---

# 15. Why the Manual Implementation Is Important

Implementing the models from scratch helps understand what machine learning libraries perform internally.

The manual implementation demonstrates the mathematical steps behind:

* Data preprocessing
* Feature encoding
* Feature scaling
* Linear Regression
* Sigmoid function
* Logistic Regression
* Gradient descent
* Model prediction
* Evaluation metrics

Instead of treating Scikit-learn models as black boxes, the implementation shows the underlying operations using NumPy and Pandas.

---

# 16. Reproducibility

The experiment uses a fixed random state:

```text
random_state = 42
```

The same train-test indices are reused for:

* Scikit-learn Linear Regression
* Manual Linear Regression
* Scikit-learn Logistic Regression
* Manual Logistic Regression
* Optimized Manual Logistic Regression

This ensures that the model comparisons are performed using the same data.

---

# 17. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* GitHub

---

# 18. Project Structure

A suggested project structure is:

```text
DS605-Lab-Assignment-5/
│
├── garment_worker_productivity.csv
├── DS605_Lab_Assignment_5.ipynb
├── README.md
└── requirements.txt
```

---

# 19. Requirements

The main Python libraries required are:

```text
numpy
pandas
scikit-learn
jupyter
```

They can be installed using:

```bash
pip install numpy pandas scikit-learn jupyter
```

---

# 20. How to Run

### Step 1 — Clone the Repository

Clone the GitHub repository to your computer.

### Step 2 — Install Dependencies

Run:

```bash
pip install numpy pandas scikit-learn jupyter
```

### Step 3 — Open the Notebook

Open:

```text
DS605_Lab_Assignment_5.ipynb
```

using Jupyter Notebook or VS Code.

### Step 4 — Add the Dataset

Make sure the dataset file is located in the same directory as the notebook:

```text
garment_worker_productivity.csv
```

### Step 5 — Run the Notebook

Run the cells from beginning to end.

The notebook will:

1. Load the dataset
2. Clean the data
3. Create the classification target
4. Split the data
5. Preprocess the features
6. Train Scikit-learn models
7. Implement models manually
8. Calculate evaluation metrics
9. Compare execution times
10. Optimize the manual Logistic Regression model

---

# 21. Conclusion

This project demonstrates the implementation and comparison of machine learning models using both Scikit-learn and manual NumPy/Pandas implementations.

The manually implemented Linear Regression reproduced the Scikit-learn regression results exactly. Similarly, the original manual Logistic Regression produced the same classification metrics as the Scikit-learn implementation.

The manual Logistic Regression was then optimized using a different learning rate, additional iterations, and L2 regularization. The optimized model improved accuracy, recall, and F1-score, although its training time increased.

Overall, the project provides practical understanding of both the use of machine learning libraries and the mathematical operations that take place internally when training regression and classification models.
