# Comprehensive Machine Learning Pipeline on Heart Disease UCI Dataset

## 1. Project Overview
This project aims to analyze, predict, and visualize **heart disease risks** using end-to-end **machine learning techniques**.  
It covers the full ML lifecycle from **data preprocessing and feature engineering** to **model training, evaluation, tuning, and deployment**.

## 2. Objectives
- Perform **Data Cleaning & Preprocessing** (missing values, encoding, scaling)
- Apply **Dimensionality Reduction (PCA)**
- Conduct **Feature Selection** using RFE, Chi-Square, and Feature Importance
- Build **Supervised Learning Models**:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - SVM  
- Perform **Unsupervised Learning**:
  - K-Means Clustering  
  - Hierarchical Clustering  
- Conduct **Hyperparameter Optimization** using GridSearchCV & RandomizedSearchCV
- **Export & Deploy Models** (`.pkl` format)

## 3. Workflow Summary
### **Step 1 – Data Preprocessing**
- Handle missing values, encode categorical columns, and scale features.
- Perform EDA (histograms, boxplots, heatmaps).

### **Step 2 – PCA (Dimensionality Reduction)**
- Reduce dimensionality while maintaining variance.
- Visualize explained variance and principal components.

### **Step 3 – Feature Selection**
- Use RFE, Feature Importance, and Chi-Square to select the most relevant features.

### **Step 4 – Supervised Learning**
- Train classification models and evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

### **Step 5 – Unsupervised Learning**
- Apply K-Means and Hierarchical Clustering.
- Compare cluster assignments to actual labels.

### **Step 6 – Hyperparameter Tuning**
- Optimize model hyperparameters using GridSearchCV and RandomizedSearchCV.

### **Step 7 – Model Export**
- Save the final optimized model as `Final_model.pkl`.

## 4. Installation & Usage
### **Step 1 – Clone the Repository**
```Bash
git clone https://github.com/maryemmohamed9/heart-disease-UCI.git
cd heart-disease-UCI
```
### **Step 2 – Install Dependencies**
```Bash
pip install -r requirements.txt
```
#### **Step 3 – Run Notebooks**
Open and execute notebooks in order (01 to 06).
