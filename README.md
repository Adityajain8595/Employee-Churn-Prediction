# 🧠 Employee Churn Prediction using Decision Tree

## 📘 Overview

This project focuses on predicting **employee attrition (churn)** using structured HR data. A XGBoost Classifier is used to model the likelihood of an employee leaving the company, based on various features such as satisfaction level, average monthly hours, promotion history, department, salary level and more.

To enhance model accuracy and prevent overfitting, the pipeline includes **feature encoding**, **standardization**, and **hyperparameter tuning** using **RandomizedSearchCV**.

---

## 🎯 Problem Statement

> Employee attrition poses a critical challenge to organizational stability and growth. This project aims to develop a machine learning model that predicts whether an employee is likely to leave the company based on HR-related numerical and categorical features. The objective is to support HR in proactively addressing employee dissatisfaction and improving retention strategies.

---

## 🗃️ Dataset

- Contains features like:
  - `satisfaction_level`
  - `last_evaluation`
  - `number_project`
  - `average_monthly_hours`
  - `time_spend_company`
  - `work_accident`
  - `promotion_last_5years`
  - `salary` (categorical)
  - `department` (categorical)

- Target variable:
  - `left` (1 = employee left, 0 = employee stayed)

---

## 🧪 Model Pipeline

1. **EDA**
   - Data Preprocessing and Cleaning
   - Made key insights about the characteristics of data
   - Visualized relationships between features and identified key factors influencing employee churn 
     
1. **Preprocessing**
   - One-Hot Encoding for `department` and `salary`
   - Standardization for numerical features

2. **Model Training**
   - Trained 5+ classification models on the processed data
   - Evaluated models on metrics such as accuracy, precision, recall, f1 score etc.
   - Selected the stand-out models in terms of f1 score and later did hyperparameter tuning 
   
3. **Hyperparameter Tuning**
   - Performed using : `RandomizedSearchCV`: faster search over parameter distributions
   - Optimized the model and selected XGBClassifier
---


## 📊 Performance Metrics (Similar for both)

- **Accuracy**: 98%
- **Precision**: 98% (for both classes)
- **Recall**:
  - Class 0 (Stayed): 99%
  - Class 1 (Left): 91%
- **F1 Score**:
  - Class 0: 99%
  - Class 1: 94%

> Metrics evaluated using a classification report on a test set.

---


## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🧩 How to Run

1. Clone the repository:
   
   >> git clone https://github.com/Adityajain8595/Employee-Churn-Prediction.git

   >> cd Employee-Churn-Prediction

2. Run the notebook:

   >> jupyter notebook attrition_model.ipynb

## 📌 Conclusion

This project demonstrates how structured HR data combined with a well-tuned model can effectively predict employee attrition, enabling organizations to take preventive retention actions based on data-driven insights.

👤 Author
Aditya Jain
📧 [meaditya1103@gmail.com]
🔗 [www.linkedin.com/in/adityajain8595/]
