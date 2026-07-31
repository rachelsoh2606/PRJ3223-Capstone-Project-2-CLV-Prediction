# PRJ3223-Capstone-Project-2-CLV-Prediction
# Using Machine Learning to Forecast Customer Lifetime Value for Better Customer Retention

A predictive analytics and Explainable AI (XAI) engine designed to forecast **Customer Lifetime Value (CLV)**, segment e-commerce customers into strategic value tiers, and deliver data-driven recommendations for retention marketing.

---

## 💼 Business Perspective & Executive Overview

In e-commerce and retail, organizations frequently focus on short-term indicators such as monthly sales figures or immediate customer acquisition counts. This short-term focus often leads to suboptimal marketing expenditures, inefficient retention prioritization, and lost opportunities for personalized engagement.

This project shifts decision-making from **reactive campaign management** to a **value-driven, proactive retention strategy**:

* **Optimized Marketing ROI**: By forecasting future CLV rather than looking solely at historical spend, marketing teams can reallocate budgets toward acquiring and retaining high-value customer profiles.
* **Differentiated Retention Framework**: Customers are automatically categorized into actionable business segments (`Premium Core`, `Mid-Tier`, `Low-Value`), allowing tailored interventions rather than a "one-size-fits-all" campaign approach.
* **Transparent AI Decisions**: Using Explainable AI (SHAP), business stakeholders can understand *why* a customer is predicted to be high- or low-value, providing clear rationale for offer personalization, VIP perks, or win-back campaigns.

---

## 📌 Abstract

Customer Lifetime Value (CLV) is a critical performance indicator for estimating long-term profitability. Traditional models (such as basic RFM frameworks or linear regressions) often fail to capture complex non-linear behavioral dynamics and feature interactions. 

This project implements an end-to-end machine learning pipeline that preprocesses transactional data, handles missingness and financial outliers, eliminates severe multicollinearity via recursive Variance Inflation Factor (VIF) pruning, and evaluates multiple regression algorithms. Using SHapley Additive exPlanations (SHAP) and K-Means clustering with PCA coordinate projection, the system translates black-box predictions into actionable managerial retention strategies.

---

## 🎯 Research Objectives & Questions

### **Research Objectives (RO)**
1. **RO1**: To determine the key transactional, behavioral, and demographic factors that influence customer lifetime value.
2. **RO2**: To create and assess various machine learning algorithms (Linear Regression, Random Forest, XGBoost) to forecast CLV.
3. **RO3**: To evaluate model outcomes using Explainable AI (XAI) and offer data-driven business insights to enhance marketing and retention strategies.

### **Research Questions (RQ)**
* **RQ 1.1**: What customer behavioral and transactional factors have the greatest impact on CLV?
* **RQ 1.2**: Which data features show the strongest correlation with customer spending and retention patterns?
* **RQ 1.3**: How do demographic or loyalty attributes influence variations in CLV across customer segments?
* **RQ 2.1**: How can Linear Regression, Random Forest, and XGBoost be used to predict customer lifetime value?
* **RQ 2.2**: Which machine learning algorithm achieves the highest predictive accuracy and generalization performance?
* **RQ 2.3**: How does feature selection or dimensionality reduction affect predictive performance?
* **RQ 3.1**: Why is CLV prediction valuable for guiding marketing budget allocation and customer segmentation?
* **RQ 3.2**: How can Explainable AI (SHAP) techniques be applied to interpret CLV predictions?
* **RQ 3.3**: What strategic recommendations can be derived to improve customer retention and profitability?

---

## 📂 Repository Structure

```text
.
├── data/                                      # Raw customer purchasing and behavioral datasets
├── outputs/                                   # Generated evaluation metrics, CSV reports, and charts
│   ├── distributions/                         # Individual feature & target distribution plots
│   ├── shap_dependence/                       # SHAP feature interaction/dependence profile plots
│   ├── clv_distribution_by_segments.png       # Strategic CLV density plot across segments
│   ├── correlation_heatmap.png                # Pearson correlation matrix heatmap
│   ├── customer_clv_predictions_val.csv       # Validation set customer-level prediction logs
│   ├── customer_segments_pca.png              # PCA 2D K-Means cluster scatter plot
│   ├── feature_importance_global_comparison.png # Multi-method global importance chart
│   ├── feature_relationship_pairplot.png        # Pairwise structural feature relationship matrix
│   ├── final_test_performance_summary.csv     # Unseen test set evaluation metrics (R², MAE, RMSE)
│   ├── model_cv_stability_boxplot.png         # 5-fold cross-validation stability boxplot
│   ├── model_performance_summary.csv          # Validation set evaluation summary table
│   ├── model_predictions_vs_actual.png        # Predicted vs. actual CLV scatter plots
│   ├── model_residual_distributions.png       # Diagnostic residual error probability density curves
│   ├── mutual_information_plot.png            # Mutual Information feature dependency chart
│   ├── mutual_information_scores.csv          # Ranked Mutual Information dependency scores
│   ├── permutation_importance_Random Forest.png # RF permutation importance plot
│   ├── permutation_importance_XGBoost.png     # XGBoost permutation importance plot
│   ├── pipeline_results_manifest.txt          # Summary text log of complete pipeline execution
│   ├── random_forest_permutation_importance.csv # Random Forest permutation metrics table
│   ├── random_forest_permutation_plot.png     # RF permutation feature plot
│   ├── shap_summary_plot.png                  # Global SHAP beeswarm summary plot
│   ├── spearman_rankings.csv                  # Spearman rank correlation coefficients
│   ├── top_10_future_clv_targets.csv          # Top 10 high-value customer target forecasts
│   ├── vif_after_filtering.csv                # Post-pruning Variance Inflation Factor scores
│   ├── vif_before_filtering.csv               # Raw pre-pruning Variance Inflation Factor scores
│   ├── vif_table.csv                          # Full VIF diagnostic table
│   ├── xgboost_builtin_importance.csv         # Built-in XGBoost feature importance rankings
│   ├── xgboost_permutation_importance.csv     # XGBoost permutation metrics table
│   └── xgboost_permutation_plot.png           # XGBoost permutation feature plot
├── src/                                       # Core source code modules
│   ├── __init__.py                            # Package marker for modular imports
│   ├── preprocessing.py                       # Imputation, Winsorization, log scaling, and encoding
│   ├── features.py                            # Feature engineering and target construction (future_clv)
│   ├── selection.py                           # Recursive VIF, Pearson/Spearman correlation & Mutual Information
│   ├── modelling.py                           # Cross-validation, RandomizedSearchCV tuning, SHAP & evaluation
│   └── visualization.py                       # Publication-quality diagnostic, PCA clustering, and residual plots
├── .gitattributes                             # Git attribute configurations
├── .gitignore                                 # Tracks files excluded from version control
├── main.py                                    # Main pipeline orchestration script
├── output_text.py                             # Script for generating console diagnostics and text summaries
├── README.md                                  # Project documentation
└── requirements.txt                           # Required Python library dependencies
```

---

## 📑 Core Module Descriptions

| Script / Directory | Primary Role & Responsibilities |
| :--- | :--- |
| `main.py` | Main entry point that coordinates data loading, preprocessing, feature engineering, selection, model training, and visual asset generation. |
| `output_text.py` | Auxiliary script for formatting, printing, and exporting detailed text logs and summary tables for report generation. |
| `src/preprocessing.py` | Handles missing value imputation, IQR outlier detection with $1\%$ Winsorization, log1p scaling, categorical encoding (Ordinal, Binary, One-Hot), and StandardScaler normalization. |
| `src/features.py` | Constructs the dependent target variable (`future_clv`) and derives behavioral intensity metrics (`avg_purchase_value`, `engagement_adjusted_value`) while dropping target leakage attributes. |
| `src/selection.py` | Performs Pearson/Spearman correlation heatmaps, recursive VIF pruning (threshold $\le 10.0$), Mutual Information regression scoring, and Permutation Importance. |
| `src/modelling.py` | Executes 5-fold cross-validation (KFold), hyperparameter tuning (RandomizedSearchCV), evaluates performance metrics ($R^2$, $MAE$, $RMSE$), computes SHAP values (TreeExplainer), and exports predictions. |
| `src/visualization.py` | Generates distribution charts, PCA 2D K-Means cluster coordinate projections, predicted vs. actual scatter plots, residual probability density curves, and SHAP dependence plots. |

---

## 📊 Experimental Results & Model Performance

The machine learning models were evaluated on an independent, unseen test dataset ($10\%$ split):

| Machine Learning Model | R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Performance Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 0.6982 | 5,204.69 | 6,739.18 | Struggled with non-linear behavioral relationships. |
| **Optimized Random Forest** | 0.9996 | 10.98 | 244.82 | Exceptional fit; slight variance in high-value tails. |
| **Optimized XGBoost** | **0.9999** | **71.12** | **121.45** | **Optimal Performer**: Highest accuracy & lowest error variance. |

---

## 📈 Strategic Customer Segmentation & Actionable Retention Framework

Using K-Means clustering ($k=3$) projected via 2D PCA, customer profiles are mapped directly to business strategies:

```text
                                  ┌───────────────────────────┐
                                  │   CLV Prediction Engine   │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌───────────────────────────┐          ┌───────────────────────────┐          ┌───────────────────────────┐
│ Premium High-Value Core   │          │    Mid-Tier Occasional    │          │   Low-Value / Inactive    │
├───────────────────────────┤          ├───────────────────────────┤          ├───────────────────────────┤
│ • 31.14% Portfolio Share  │          │ • 47.15% Portfolio Share  │          │ • 21.71% Portfolio Share  │
│ • High Engagement         │          │ • High Spending Power     │          │ • Low Engagement          │
│ • Strong Loyalty Status   │          │ • Lower Purchase Freq.    │          │ • Low Retention Rate      │
├───────────────────────────┤          ├───────────────────────────┤          ├───────────────────────────┤
│    STRATEGY: RETAIN       │          │      STRATEGY: GROW       │          │   STRATEGY: REACTIVATE    │
├───────────────────────────┤          ├───────────────────────────┤          ├───────────────────────────┤
│ • VIP Reward Program      │          │ • Upselling/ Cross-selling│          │ • Automated Email Sequence│
│ • Dedicated Concierge     │          │ • Loyalty Enrollment      │          │ • Targeted Win-back Offers│
│ • Early Access Perks      │          │ • Personalized Offers     │          │ • Low-cost Outreach       │
└───────────────────────────┘          └───────────────────────────┘          └───────────────────────────┘
```

---

## ⚡ Quick Start & Setup

### 1. Environment Setup
Clone the repository, setup virtual environment, and install dependencies:
```bash
git clone https://github.com/rachelsoh2606/PRJ3223-Capstone-Project-2-CLV-Prediction.git
cd PRJ3223-Capstone-Project-2-CLV-Prediction

# Create a virtual environment
python -m venv venv

# activate virtual environment
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your dataset is placed inside the `data/` directory named `customer_data.csv`.

### 3. Run Pipeline
Execute the complete end-to-end analytical pipeline:
```bash
python main.py
```

---

## 🛠️ Tech Stack & Dependencies

* **Core Runtime**: Python 3.12
* **Data Processing & Numerical Analysis**: `numpy==2.4.6`, `pandas==3.0.5`, `scipy==1.18.0`
* **Machine Learning & Statistical Modeling**: `scikit-learn==1.9.0`, `xgboost==3.3.0`, `statsmodels==0.14.6`, `patsy==1.0.2`, `numba==0.66.0`
* **Explainable AI (XAI)**: `shap==0.52.0`, `slicer==0.0.8`, `cloudpickle==3.1.2`
* **Visualization & Utilities**: `matplotlib==3.11.1`, `seaborn==0.13.2`, `pillow==12.3.0`, `openpyxl==3.1.5`, `tqdm==4.70.0`, `joblib==1.5.3`
