# Customer Lifetime Value (CLV) Prediction & Analytics Engine

A modular machine learning pipeline built to forecast **Future Customer Lifetime Value (CLV)**, segment e-commerce customers, and deliver Explainable AI (XAI) diagnostics for targeted retention marketing.

---

## 📌 What These Python Files Do

This project is modularized into distinct Python scripts inside the `src/` directory. Each file handles a specific stage of the data science and machine learning lifecycle:

### 1. `src/preprocessing.py` — Data Cleaning & Standardization
* **Missing Value Handling**: Eliminates records with $>50\%$ missing data, imputes numerical features (median for skewed variables, mean for symmetric ones), and mode-imputes categorical fields[cite: 3].
* **Outlier Capping & Log Transformation**: Uses IQR detection, Winsorization ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles), and `log1p` transformation on financial columns (`income`, `purchase_amount`)[cite: 3].
* **Categorical Encoding**: Implements Ordinal encoding for ordered features (e.g., `purchase_frequency`, `education`), Binary encoding (`gender`, `promotion_usage`), and One-Hot encoding (`region`, `product_category` with `drop_first=True`)[cite: 3].
* **Feature Scaling**: Fits `StandardScaler` strictly on training data to prevent data leakage before scaling validation and test sets[cite: 3].

---

### 2. `src/features.py` — Feature Engineering & Target Formulation
* **Target Construction**: Formulates the `future_clv` target variable by combining base purchase behavior with loyalty and promotion multipliers[cite: 1]:
  $$\text{Future CLV} = (\text{Purchase Amount} \times \text{Purchase Frequency}) \times (1 + \text{Loyalty Status} \times 0.05) \times (1 + \text{Promotion Usage} \times 0.03)$$
* **Derived Behavioral Features**: Generates domain-specific metrics such as `avg_purchase_value` (spending intensity) and `engagement_adjusted_value`[cite: 1].
* **Target Leakage Prevention**: Safely drops raw input variables (`purchase_amount`, `purchase_frequency`, `income`) after feature creation to ensure models learn predictive customer characteristics rather than duplicating the formula[cite: 1].

---

### 3. `src/selection.py` — Feature Selection & Multicollinearity
* **Correlation Analysis**: Generates Pearson correlation heatmaps and computes Spearman rank correlations against `future_clv`[cite: 4].
* **Recursive VIF Pruning**: Iteratively calculates Variance Inflation Factors and drops collinear variables until all remaining features achieve $\text{VIF} \le 10.0$[cite: 4].
* **Mutual Information Analysis**: Evaluates non-linear dependencies between input features and the target variable[cite: 4].
* **Permutation Feature Importance**: Measures performance drop when individual features are shuffled across trained ensemble models[cite: 4].

---

### 4. `src/modelling.py` — Machine Learning, Evaluation & XAI
* **Model Training & Tuning**: Fits a **Linear Regression** baseline and optimizes **Random Forest** and **XGBoost Regressors** using 5-Fold Cross-Validation (`KFold`) and `RandomizedSearchCV`[cite: 2].
* **Performance Auditing**: Computes $R^2$, $MAE$, and $RMSE$ across validation and unseen test sets[cite: 2].
* **Explainable AI (XAI)**: Utilizes `shap.TreeExplainer` to extract global feature impacts and individual feature dependence profiles[cite: 2].
* **Operational Outputs**: Exports customer-level validation predictions and top high-value CLV target lists (`top_10_future_clv_targets.csv`) for retention teams[cite: 2].

---

### 5. `src/visualization.py` — Diagnostic Graphics & Plotting
* **Exploratory Plots**: Renders distribution histograms, boxplots, categorical count plots, and before/after outlier comparison grids[cite: 5].
* **Customer Segmentation**: Executes K-Means clustering ($k=3$) and projects customers onto a $2\text{D}$ Principal Component Analysis (PCA) coordinate space (`Premium High-Value Core`, `Mid-Tier Occasional`, `Low-Value / Inactive`)[cite: 5].
* **Model Evaluation Visuals**: Draws predicted vs. actual scatter plots with identity lines, probability density curves of residual errors, and cross-validation stability boxplots[cite: 5].
* **Consensus & SHAP Plots**: Plots cross-methodology feature importance matrices and custom SHAP dependence graphs[cite: 5].

---

### 6. `src/__init__.py` — Package Marker
* Marks the `src/` directory as an importable Python package, enabling clean inter-module imports across scripts[cite: 2, 6].


---

## 🛠️ Required Dependencies

To run these scripts, ensure the following Python packages are installed:

```text
Package         Version
--------------- -----------
cloudpickle     3.1.2
colorama        0.4.6
contourpy       1.3.3
cycler          0.12.1
et_xmlfile      2.0.0
fonttools       4.63.0
joblib          1.5.3
kiwisolver      1.5.0
llvmlite        0.48.0
matplotlib      3.11.1
narwhals        2.24.0
numba           0.66.0
numpy           2.4.6
openpyxl        3.1.5
packaging       26.2
pandas          3.0.5
patsy           1.0.2
pillow          12.3.0
pip             25.0.1
pyparsing       3.3.2
python-dateutil 2.9.0.post0
scikit-learn    1.9.0
scipy           1.18.0
seaborn         0.13.2
shap            0.52.0
six             1.17.0
slicer          0.0.8
statsmodels     0.14.6
threadpoolctl   3.6.0
tqdm            4.70.0
tzdata          2026.3
xgboost         3.3.0
