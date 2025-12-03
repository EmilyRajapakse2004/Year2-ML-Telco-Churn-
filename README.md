# Telco Customer Churn Prediction System

A complete **Machine Learning project** to predict whether a customer will **churn** (cancel their subscription) for a telecommunications company. This project uses both **Decision Tree** and **Neural Network** models and allows users to interactively add new customer data for prediction. All predictions are saved in a **tabular CSV file**, making it easy to analyze and share results.

---

## **Project Overview**

Customer churn prediction is critical for telecom companies to **retain customers** and reduce revenue loss. This project demonstrates the **end-to-end ML workflow**, including:

- Data collection and preprocessing  
- Exploratory Data Analysis (EDA) with visualizations  
- Feature engineering  
- Model training (Decision Tree & Neural Network)  
- Model evaluation using classification metrics and ROC-AUC  
- Interactive prediction of new customer churn  
- Saving predictions in a structured, Excel-friendly CSV  
---

## **Key Features**

- **Exploratory Data Analysis (EDA):**  
  - Distribution plots for churn, tenure, and other features  
  - Correlation heatmaps for numeric variables  

- **Data Preprocessing:**  
  - Handling missing values  
  - Label encoding for categorical variables  
  - One-hot encoding and scaling for numeric features  

- **Model Training:**  
  - Decision Tree classifier with hyperparameter tuning  
  - Neural Network (multi-layer dense) with training/validation split  

- **Evaluation Metrics:**  
  - Accuracy, precision, recall, F1-score  
  - Confusion matrix and ROC-AUC curves  

- **Interactive Customer Prediction:**  
  - Validated input prompts for categorical and numeric fields  
  - Choice between Neural Network (`nn`) or Decision Tree (`dt`) model  
  - Predictions saved in **tabular CSV** (`results/new_customers.csv`)  

- **User-Friendly CSV Export:**  
  - Column headers included  
  - Overwrites old entries at runtime  
  - Appends multiple new customers in the same session  

---


## Directory Structure

```
.
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── EDA.ipynb
├── results/
│   ├── decision_tree_model.pkl
│   ├── neural_network_model.keras
│   └── new_customers.csv
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── predict.py
├── main.py
├── requirements.txt

```


