# Telco Customer Churn Prediction – Data Science Capstone

## Project Objective
A telecom company loses revenue when customers leave. This project builds machine learning models to **predict which customers are likely to churn**, so a retention team can intervene early. It covers the full workflow: data cleaning, EDA, feature engineering, model training/tuning, evaluation, and business recommendations.

**Success metrics:** F1-score, recall on the churn class, and ROC-AUC (accuracy alone is misleading because only ~27% of customers churn).

## Dataset
- **Name:** IBM Telco Customer Churn
- **Source:** Kaggle – https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size:** 7,043 customers, 21 columns (demographics, account info, subscribed services, charges, churn label)
- Place the file at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── data/                       # raw dataset (Telco churn CSV)
├── notebooks/
│   └── capstone_analysis.ipynb # EDA, modeling, evaluation
├── src/
│   └── data_preprocessing.py   # cleaning, feature engineering, encoding, scaling
└── reports/
    ├── final_report.pdf
    ├── presentation_slides.pdf
    ├── model_comparison.csv
    └── figures/                # charts used in the report and slides
```

## Installation
```bash
git clone https://github.com/sarkarsoniya8-cmyk/data-science-capstone.git
cd data-science-capstone
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
1. Download the dataset into `data/`.
2. Test the preprocessing: `python src/data_preprocessing.py`
3. Open and run all cells in `notebooks/capstone_analysis.ipynb` (also works in Google Colab: upload the repo, set the working directory to the repo root).

## Methodology
1. **Cleaning:** fixed `TotalCharges` (text with blanks for new customers), removed ID column and duplicates, merged "No internet/phone service" into "No".
2. **Feature engineering:** `num_addons`, `avg_monthly_spend`, `tenure_group`.
3. **Preprocessing:** binary and one-hot encoding, IQR outlier capping, standard scaling. Outlier bounds and scaler are fit on the training set only (no data leakage).
4. **Models:** Logistic Regression (baseline) and Random Forest tuned with `RandomizedSearchCV` (5-fold stratified CV, F1 scoring), both with balanced class weights.
5. **Evaluation:** classification report, confusion matrix, ROC curve, feature importance on a stratified 80/20 hold-out set.

## Key Findings
> Fill in after running the notebook with your actual numbers.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | _ | _ | _ | _ | _ |
| Random Forest (tuned) | _ | _ | _ | _ | _ |

- Month-to-month contracts and short tenure are the strongest churn signals.
- Higher monthly charges and fiber-optic service are associated with higher churn.
- Customers with add-on services (tech support, online security) churn less.

## Business Recommendations
1. Incentivize month-to-month customers to move to 1–2 year contracts.
2. Build an onboarding/retention programme for the first 12 months.
3. Bundle support and security add-ons to increase stickiness.
4. Investigate fiber-optic pricing/quality and electronic-check payment friction.

## Limitations & Future Work
- Single snapshot of data, no time dimension; class imbalance; no retention-cost data.
- Future: XGBoost/LightGBM, probability calibration, cost-based threshold tuning, SHAP explanations.

## Deliverables
- Detailed report: `reports/final_report.pdf`
- Slide deck: `reports/presentation_slides.pdf`
