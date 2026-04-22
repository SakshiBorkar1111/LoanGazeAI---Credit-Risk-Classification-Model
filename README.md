# LoanGazeAI---Credit-Risk-Classification-Model
End-to-end credit risk classification model to automate loan approval prediction using 8 ML algorithms, GridSearchCV tuning, and a reusable inference pipeline. Built with Python, Scikit-learn &amp; XGBoost.

---

## 📌 Problem Statement

**Dream Housing Finance** provides home loans across urban, semi-urban, and rural areas. Currently, loan eligibility is assessed manually — a slow and inconsistent process.

**Goal:** Build a machine learning model to automate real-time loan eligibility prediction based on applicant details filled in an online application form. This enables the company to instantly identify eligible customer segments and reduce manual underwriting effort.

---

##  Business Relevance

This project is directly applicable to:
- **Credit Risk Assessment** — predicting default likelihood before loan disbursement
- **Insurance Underwriting** — profiling applicants based on financial and demographic risk factors
- **NBFC / Banking Operations** — automating the pre-screening stage of loan processing

---

## 📂 Dataset Features

| Feature | Description |
|---|---|
| Gender | Applicant's gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Employment type |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Requested loan amount (in thousands) |
| Loan_Amount_Term | Repayment term (in months) |
| Credit_History | Good / Bad credit track record |
| Property_Area | Rural / Semiurban / Urban |
| **Loan_Status** |  Target Variable — Approved (Y) / Rejected (N) |

---

## 🔬 Project Workflow

```
1. Business Understanding
        ↓
2. Data Understanding & EDA
        ↓
3. Data Preparation (Cleaning + Wrangling)
        ↓
4. Modelling & Evaluation
        ↓
5. Model Saving & Inference Pipeline
```

---

## 📊 Exploratory Data Analysis (EDA)

- **Univariate Analysis:** Histograms for continuous variables; count plots for categorical variables
- **Bivariate Analysis:** Crosstabs between `Loan_Status` and each categorical variable (Marriage, Education, Employment, Property Area)
- **Correlation Heatmap:** Income and LoanAmount show ~62% positive correlation — higher income → higher loan amount
- **Distribution Finding:** Both Income and LoanAmount are heavily right-skewed

---

## 🛠️ Data Preparation

### Cleaning
- Replaced `3+` in Dependents with integer `3`
- Missing value treatment:
  - Categorical columns → filled with **mode**
  - Critical numerical columns (Income, LoanAmount, Term, Credit_History) → **rows dropped** (cannot impute loan-critical features)
- Data type corrections for `Dependents` and `Loan_Amount_Term`

### Feature Engineering
- Created combined `Income` = `ApplicantIncome` + `CoapplicantIncome`
- Dropped the original split income columns

### Encoding
- Binary encoding for Gender, Married, Education, Self_Employed, Credit_History, Loan_Status
- Ordinal encoding for Property_Area (Rural=0, Semiurban=1, Urban=2)

### Transformation
- Applied **Box-Cox transformation** on Income and LoanAmount to correct right skew → improved model stability

---

## 🤖 Models Trained

| # | Model | Tuning Method |
|---|---|---|
| 1 | Logistic Regression | Default |
| 2 | K-Nearest Neighbors | GridSearchCV (n_neighbors) |
| 3 | Support Vector Machine | GridSearchCV (C, kernel) |
| 4 | Decision Tree | GridSearchCV (criterion, max_depth) |
| 5 | Random Forest | GridSearchCV (n_estimators) |
| 6 | AdaBoost | GridSearchCV (n_estimators) |
| 7 | Gradient Boosting | GridSearchCV (n_estimators, learning_rate) |
| 8 | XGBoost | GridSearchCV (n_estimators, max_depth, gamma) |

**Evaluation Metrics Used:** Training Accuracy, 5-Fold Cross-Validation Score, Test Accuracy

**Feature Selection:** Ensemble-based feature importance scores were used to identify and retain only significant predictors for each tree-based model.

---

## 💾 Model Saving & Inference

The best-performing model (Decision Tree with tuned hyperparameters) was serialised using `joblib`:

```python
from joblib import dump
dump(dt, 'loan.joblib')
```

A complete **inference pipeline** was built to process new applicant data — applying the same preprocessing steps (encoding, transformation, feature selection) before generating a prediction.

```python
# Sample prediction output
dt.predict(X_new)
# Output: [0] → Not Eligible  |  [1] → Eligible

---


## 🔮 Future Scope

- Deploy the model as a REST API using Flask or FastAPI
- Add SHAP-based explainability to interpret individual predictions
- Experiment with class imbalance techniques (SMOTE) to improve minority class recall
- Integrate a front-end form for real-time predictions

---



