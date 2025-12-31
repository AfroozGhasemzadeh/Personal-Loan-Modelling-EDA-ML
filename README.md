# Personal-Loan-Modelling-EDA-ML
#  Personal Loan Modelling — Machine Learning Classification Project
This project focuses on predicting whether a customer will accept a personal loan offer using machine‑learning techniques. The workflow includes data cleaning, exploratory data analysis (EDA), model comparison, hyperparameter tuning, and final model selection. All visualizations generated throughout the analysis are saved in the folder Personal Loan Modelling Images.
#  Dataset
The dataset is sourced from Kaggle: /kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv
It contains 1000 customer records with the following features:

| Feature              | Description                                                   |
|----------------------|---------------------------------------------------------------|
| **ID**               | Unique customer identifier                                    |
| **Age**              | Customer age                                                  |
| **Experience**       | Years of professional experience                              |
| **Income**           | Annual income (in thousands)                                  |
| **ZIP Code**         | Residential ZIP code                                          |
| **Family**           | Family size                                                   |
| **CCAvg**            | Average monthly credit card spending                          |
| **Education**        | Education level (1 = Undergrad, 2 = Graduate, 3 = Advanced)   |
| **Mortgage**         | Mortgage value                                                |
| **Personal Loan**    | Target variable (1 = accepted, 0 = not accepted)              |
| **Securities Account** | Whether the customer holds a securities account             |
| **CD Account**       | Whether the customer holds a certificate of deposit           |
| **Online**           | Uses online banking                                           |
| **CreditCard**       | Has a credit card issued by the bank                          |
#  Data Cleaning & Preprocessing
1. Initial Visualization & Data Description
The notebook begins with:
- Summary statistics
- Distribution plots
- Countplots for categorical variables
- Boxplots for numerical variables
This helps identify unusual patterns and potential data issues.
2. Correcting Illogical Values
During inspection, several inconsistencies were found — for example:
- Negative values in the Experience column, which are not logically possible.
These values were corrected using reasonable assumptions based on the distribution of the data.
3. Outlier Detection & Removal
Outliers were examined using:
- Boxplots
- Distribution plots
- Domain knowledge
Only necessary outliers were removed to maintain data integrity.
# Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Customer demographics
- Spending behavior
- Loan acceptance patterns
- Relationships between features
This step provides intuition before modelling and helps guide feature selection.
# Model Building & Comparison
Multiple machine‑learning algorithms were trained and evaluated:
- Logistic Regression
- K‑Nearest Neighbors (KNN)
- Naive Bayes
Each model was assessed using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve & AUC
- Log Loss
- Confusion Matrix
Cross‑validation (Stratified K‑Fold) was also applied to ensure stability
# Best Model: KNN (k = 3)
After comparing all models, KNN with k = 3 achieved the strongest performance across key metrics and was selected as the final model
# New Data Prediction
To demonstrate practical use, a new customer record was manually created and fed into the final KNN model to predict whether this customer would accept a personal loan.
This step shows how the model can be applied in a real‑world scenario.
# Repository Structure
├── Personal Loan Modelling.ipynb
├── Personal Loan Modelling Images/
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── knn_accuracy_vs_k.png
│   ├── knn_f1_score_vs_k.png
│   ├── knn_auc_vs_k.png
│   ├── cross_validation_plots.png
│   └── other visualizations...
├── README.md











