
#  Credit Card Fraud Detection Using Machine Learning

A complete machine learning pipeline to detect fraudulent credit card transactions using classical ML algorithms. This project demonstrates the full workflow — from Exploratory Data Analysis (EDA) and preprocessing to model training, evaluation, and comparison — all implemented in a single Jupyter Notebook.

🔗 **Live Notebook:** *(https://github.com/ammakollaPrasanna/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/main/Credit%20card%20fraud%20detetion%20mini%20project-1%20(1).ipynb)*

---

## ✨ Key Features

* 🔍 **Exploratory Data Analysis (EDA)**
  Includes class distribution plots, fraud vs normal comparisons, and a correlation heatmap.

* ⚖️ **Class Imbalance Handling**
  Uses stratified splitting and class-weight balancing for skewed datasets.

* 📐 **Feature Scaling**
  Applies StandardScaler correctly to prevent data leakage.

* 🤖 **Multiple ML Models**

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (RBF Kernel)

* 📊 **Evaluation Metrics**
  Accuracy, Precision, Recall, F1-score, and Confusion Matrix visualizations.

* 📈 **Model Comparison**
  Grouped bar charts comparing all models side-by-side.

---

## 🏆 Model Performance

| Model               | Accuracy | Precision | Recall  | F1 Score |
| ------------------- | -------- | --------- | ------- | -------- |
| Logistic Regression | 99.72%   | 99.71%    | 99.74%  | 99.72%   |
| Random Forest       | 99.98%   | 99.99%    | 99.97%  | 99.98%   |
| SVM (RBF)           | 100.00%  | 100.00%   | 100.00% | 100.00%  |

---

## 📊 Visualizations Included

* Class Distribution (Bar Chart & Pie Chart)
* Transaction Amount Analysis
* Correlation Heatmap
* Confusion Matrix Heatmaps
* Training History Plots
* Model Comparison Charts

---

## 🧠 Technology Stack

| Layer         | Tools Used                        |
| ------------- | --------------------------------- |
| Data Handling | Pandas, NumPy                     |
| Visualization | Matplotlib, Seaborn               |
| Preprocessing | StandardScaler, Stratified Splits |
| Models        | Scikit-learn (LR, RF, SVM)        |
| Environment   | Jupyter Notebook / Google Colab   |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ammakollaprasanna/credit-card-fraud-detection

# Navigate into project
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset

This project uses the **Credit Card Fraud Detection 2023 dataset**.

* File: `creditcard_2023.csv`
* Source: Kaggle

📌 Place the dataset in:

```
MyDrive/Dataset/creditcard_2023.csv
```

(or update the path in the notebook)

---

## ▶️ Run the Project

### Option 1 — Jupyter Notebook

```bash
jupyter notebook Credit_card_fraud_detection_mini_project.ipynb
```

### Option 2 — Google Colab

* Upload the notebook
* Mount Google Drive
* Run all cells

---

## 🔄 Pipeline Overview

```
Raw Data
   ↓
EDA & Visualization
   ↓
Preprocessing (Scaling + Stratified Split)
   ↓
Model Training (LR, RF, SVM)
   ↓
Evaluation (Metrics + Confusion Matrix)
   ↓
Comparison (Charts)
```

---

## ⚙️ Model Details

### Logistic Regression

* Baseline linear model
* Uses Stratified K-Fold Cross Validation
* Handles imbalance with class weights

### Random Forest

* Ensemble of decision trees
* Parallel processing (n_jobs=-1)
* High accuracy and robustness

### SVM (RBF Kernel)

* Non-linear classifier
* Uses `class_weight='balanced'`
* Achieves top performance

---

## 📋 Dataset Format

| Column | Description                    |
| ------ | ------------------------------ |
| V1–V28 | PCA-transformed features       |
| Amount | Transaction amount             |
| Class  | Target (0 = Normal, 1 = Fraud) |

---

## 🤝 Contributing

Contributions are welcome! You can improve this project by:

* Adding SMOTE / resampling techniques
* Implementing XGBoost / LightGBM
* Hyperparameter tuning
* Adding SHAP explainability
* Deploying as a web application

---

## 📄 License

This project is licensed under the **MIT License**.
