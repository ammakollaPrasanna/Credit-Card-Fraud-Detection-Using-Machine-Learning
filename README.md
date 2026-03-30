# Credit-Card-Fraud-Detection-Using-Machine-Learning
Credit Card Fraud Detection — Mini Project
A machine learning pipeline that detects fraudulent credit card transactions using classical ML algorithms. Built with scikit-learn, it covers the full workflow — from EDA and preprocessing to model training, evaluation, and comparison — all inside a single Jupyter Notebook.
Live Notebook: (link to your Colab / Kaggle / GitHub)

✨ Key Features
🔍 Exploratory Data Analysis — Class distribution bar charts, pie charts, fraud vs. normal transaction amount comparisons, and a full correlation heatmap
⚖️ Class Imbalance Handling — Stratified splits and class-weight computation to deal with the heavily skewed fraud/normal ratio
📐 Feature Scaling — StandardScaler applied correctly — fit on train, transform on val/test — to prevent data leakage
🤖 Three ML Models — Logistic Regression, Random Forest (parallelized), and SVM (RBF kernel with balanced class weights)
📊 Rich Evaluation Suite — Confusion matrices (heatmap), accuracy, precision, recall, F1 score, and a grouped bar chart comparing all models side by side
📈 Training History Tracking — Loss, precision, recall, false negatives, and validation metrics plotted across folds using StratifiedKFold

🏆 Model Performance
ModelAccuracyPrecisionRecallF1 ScoreLogistic Regression99.72%99.71%99.74%99.72%Random Forest99.98%99.99%99.97%99.98%SVM100.00%100.00%100.00%100.00%

Confusion matrix breakdown (on test set, ~113,726 transactions):
ModelTNFPFNTPLogistic Regression56,70016315056,713Random Forest56,85941856,845SVM56,8630056,863


📊 EDA & Visualizations
The notebook includes these visual analyses out of the box:

Class Distribution Bar Chart — Frequency of Normal vs. Fraud transactions
Pie Chart — Percentage breakdown of Fraudulent vs. Normal
Amount Statistics Table — describe() side-by-side for fraud and normal subsets
Correlation Heatmap — Seismic colormap across all 30 PCA-transformed features + Amount
Training History Plots — 8-panel subplot grid tracking Loss, FN, Precision, and Recall across folds
Confusion Matrix Heatmaps — One per model, seaborn annotated
Grouped Bar Chart — All 4 metrics × 3 models in a single comparison figure


🧠 Technology Stack
LayerToolsData HandlingPandas, NumPyVisualisationMatplotlib, SeabornPreprocessingStandardScaler, Stratified splitsModelsScikit-learn — LogisticRegression, RandomForestClassifier, SVCEvaluationclassification_report, confusion_matrix, f1_score, accuracy_scoreNotebook EnvironmentGoogle Colab (Python 3)

🚀 Getting Started
Prerequisites

Python 3.8+
pip

Installation
bash# 1. Clone the repository
git clone https://github.com/ammakollaprasanna/credit-card-fraud-detection
cd credit-card-fraud-detection

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
Dataset
The notebook expects the Credit Card Fraud Detection 2023 dataset (available on Kaggle):
creditcard_2023.csv

Place it in your Google Drive at MyDrive/Dataset/creditcard_2023.csv, or update the path at the top of the notebook to your local path.

Run
bash# Option A — Jupyter locally
jupyter notebook Credit_card_fraud_detection_mini_project.ipynb

# Option B — Google Colab
# Upload the notebook and mount your Drive, then run all cells

🗂️ Project Structure
credit-card-fraud-detection/
├── Credit_card_fraud_detection_mini_project.ipynb   # Main notebook — EDA + models
├── README.md                                         # You are here
└── requirements.txt                                  # Python dependencies

🔄 Pipeline Overview
Raw CSV Data
    │
    ▼
EDA & Visualisation
    │  Class distribution, correlation heatmap, amount stats
    ▼
Preprocessing
    │  Drop 'Class' → X/y split
    │  StandardScaler (fit on train only)
    │  Stratified Train / Validate / Test split
    ▼
Model Training
    │  Logistic Regression  (class_weight via manual weights)
    │  Random Forest        (n_estimators=50, n_jobs=-1)
    │  SVM                  (RBF kernel, class_weight='balanced')
    ▼
Evaluation
    │  Confusion matrices, classification report
    │  Accuracy / Precision / Recall / F1
    ▼
Comparison
    └─ Grouped bar chart across all models & metrics

⚙️ How the Models Work
Logistic Regression
A linear baseline with StratifiedKFold (5 splits) for cross-validation. Class weights are computed manually from training label frequencies (w_p, w_n) and passed as sample_weight. Training history (loss, precision, recall, FN/FP) is logged per fold and plotted.
Random Forest
An ensemble of 50 decision trees trained with full CPU parallelism (n_jobs=-1). Despite using no explicit class weighting, the tree structure naturally handles moderate imbalance. Fast and highly accurate.
SVM (RBF Kernel)
Support Vector Machine with a Radial Basis Function kernel and class_weight='balanced' to compensate for the fraud minority class. Achieves perfect scores on the test set for this dataset.

📋 Required CSV Format
The notebook expects the creditcard_2023 dataset with this structure:
ColumnDescriptionV1 – V28PCA-transformed anonymised featuresAmountTransaction amountClassTarget label — 0 = Normal, 1 = Fraud

🤝 Contributing
Pull requests are welcome! Some areas that could use help:

SMOTE / undersampling experiments to further address class imbalance
XGBoost / LightGBM model additions
Threshold tuning to optimise for Recall (minimise false negatives)
SHAP explainability for feature importance
Deployment as a Flask / FastAPI inference endpoint


📄 License
MIT — use it, learn from it, improve it.

Built with Scikit-learn · Pandas · Matplotlib · Seaborn · Google Colab
