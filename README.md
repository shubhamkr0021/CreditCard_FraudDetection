# 💳 Credit Card Fraud Detection using Machine Learning

## Project Overview
This project focuses on detecting **fraudulent credit card transactions** using **supervised machine learning**.  
Due to the highly **imbalanced dataset** (fraud cases are very rare compared to legitimate transactions), special attention was given to handling data imbalance, model evaluation metrics, and feature engineering.

The objective was to build a **robust classification model** that accurately identifies fraudulent activities while minimizing false positives.

---

## Objectives
- Understand and analyze transaction data distribution.  
- Handle **class imbalance** effectively using resampling techniques (SMOTE).  
- Compare multiple **machine learning algorithms** for fraud detection.  
- Evaluate model performance using metrics beyond accuracy — **Precision**, **Recall**, and **F1-score**.  
- Provide data-driven insights to improve fraud monitoring systems.

---

## Dataset Information
The dataset used is from the **Kaggle Credit Card Fraud Detection Dataset**.  
It contains transactions made by European cardholders in **September 2013**.

| Feature | Description |
|----------|-------------|
| Time | Seconds elapsed between transactions |
| V1–V28 | Anonymized features (PCA transformed) |
| Amount | Transaction amount |
| Class | Target variable — `1` for Fraud, `0` for Legitimate |

- **Total Records:** 284,807  
- **Fraudulent Transactions:** 492  
- **Imbalance Ratio:** 0.172%

---

## Exploratory Data Analysis (EDA)

### 🔹 1. Data Distribution
- Legitimate transactions dominate the dataset (>99.8%).  
- Fraud transactions are rare and scattered.

### 🔹 2. Correlation Analysis
- PCA components (`V1–V28`) show minimal direct correlation with the target variable.  
- `Amount` and `Time` are not normalized — scaling improves performance.

### 🔹 3. Class Imbalance Handling
To counter imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to oversample fraudulent transactions, balancing the dataset for training.

---

## ⚙️ Methodology

| Step | Description |
|------|--------------|
| Data Preprocessing | Scaled numerical features using StandardScaler |
| Train-Test Split | 80:20 ratio |
| Resampling | Applied SMOTE for class balance |
| Model Training | Logistic Regression, Decision Tree, Random Forest, SVM |
| Model Evaluation | Confusion Matrix, ROC-AUC, Precision, Recall, F1-score |

---

## Tools & Libraries
`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Scikit-learn` • `Imbalanced-learn`

---

##  Models Implemented

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|-----------|------------|----------|-----------|-----------|
| Logistic Regression | 98.9% | 0.86 | 0.82 | 0.84 | 0.97 |
| Decision Tree | 99.1% | 0.89 | 0.87 | 0.88 | 0.98 |
| Random Forest | **99.7%** | **0.95** | **0.93** | **0.94** | **0.99** |
| Support Vector Machine | 98.5% | 0.83 | 0.79 | 0.81 | 0.96 |

 **Random Forest** achieved the best balance between accuracy and recall, making it the most reliable model for detecting fraudulent transactions.

---

## Visualization Highlights

### 🔹 Transaction Amount vs Class
Fraudulent transactions tend to involve **smaller and irregular amounts**, unlike legitimate ones.

### 🔹 Correlation Heatmap
Most PCA-transformed features are **uncorrelated**, confirming data anonymization.

### 🔹 Confusion Matrix
Showed the trade-off between **false positives (legit marked as fraud)** and **false negatives (fraud missed)**.

### 🔹 ROC Curve
Random Forest and Decision Tree achieved **AUC > 0.98**, confirming excellent classification ability.

---

## Evaluation Metrics

Accuracy alone is misleading in imbalanced datasets — therefore:
- **Precision:** Measures how many predicted frauds are actual frauds.  
- **Recall:** Measures how many actual frauds were correctly detected.  
- **F1-score:** Balances precision and recall for holistic performance.  
- **ROC-AUC:** Evaluates overall discrimination power of the classifier.

---

## Key Insights
- High imbalance in transaction data requires **resampling or cost-sensitive learning**.  
- **Random Forest** and **Decision Tree** outperform linear models due to non-linear feature interactions.  
- **Precision-Recall trade-off** is crucial — missing frauds (false negatives) can be more costly than false alerts.  
- Continuous retraining with new data can improve detection accuracy in production.

---

## Conclusion
The project successfully demonstrated that **machine learning can detect fraudulent activities** with high reliability.  
By using robust preprocessing, balanced sampling, and ensemble models, false positives were reduced while maintaining high recall.

A **Random Forest model** emerged as the optimal classifier with **99.7% accuracy and 0.99 AUC**, suitable for deployment in fraud monitoring systems.

---

## 📜 Acknowledgement
Dataset Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

