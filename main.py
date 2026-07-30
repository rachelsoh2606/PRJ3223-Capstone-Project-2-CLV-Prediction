"""
Main Execution Pipeline

This script integrates all methodological components:
1. Dataset loading
2. Data preprocessing
3. Feature engineering
4. Feature selection
5. Dataset partitioning
6. Machine learning modelling
7. Model interpretation

The workflow follows the methodology described in Chapter 3.
"""


# ==============================================================================
# 1. IMPORT REQUIRED LIBRARIES AND MODULES
# ==============================================================================

import os

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive"
)

import pandas as pd
from sklearn.model_selection import train_test_split


# Data processing modules
from src.preprocessing import clean_data, scale_features
from src.features import engineer_features


# Feature selection modules
from src.selection import (
    run_correlation_analysis,
    calculate_vif_recursive,
    run_mutual_information
)


# Visualization modules
from src.visualization import (
    plot_feature_distributions,
    plot_categorical_distributions,
    plot_outlier_treatment_comparison,
    plot_target_distribution,
    plot_feature_pairplot,
    plot_customer_segments_2d,
    plot_clv_segment_distributions
)


# Machine learning pipeline
from src.modelling import run_machine_learning_pipeline

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/distributions", exist_ok=True)
os.makedirs("outputs/shap_dependence", exist_ok=True)

# ==============================================================================
# 2. DATA COLLECTION AND LOADING (SECTION 3.2.1)
# ==============================================================================

# Load customer behaviour dataset collected from Kaggle
df = pd.read_csv(
    "data/customer_data.csv"
)



# ==============================================================================
# 3. DATA PREPROCESSING (SECTION 3.3)
# ==============================================================================

# Apply preprocessing pipeline:
# - Missing value handling
# - Outlier treatment
# - Feature transformation
# - Categorical encoding

df_cleaned = clean_data(df)



# ==============================================================================
# 4. FEATURE ENGINEERING AND TARGET GENERATION (SECTION 3.4 & 3.5)
# ==============================================================================

# Generate engineered customer behavioural features
# and calculate future customer lifetime value (Future CLV)

df_final = engineer_features(df_cleaned)



# ==============================================================================
# 5. EXPLORATORY DATA ANALYSIS VISUALIZATION (SECTION 3.3.2 & 3.3.3)
# ==============================================================================

# Numerical feature distribution analysis
numerical_features = [
    'age',
    'satisfaction_score',
    'log_purchase_amount',
    'log_income'
]

plot_feature_distributions(
    df_final,
    numerical_features
)


# Categorical feature distribution analysis
categorical_features = [
    'gender',
    'education',
    'region',
    'loyalty_status',
    'product_category',
    'promotion_usage'
]

plot_categorical_distributions(
    df_final,
    categorical_features
)


# Compare raw and transformed distributions
# to verify outlier treatment effectiveness

plot_outlier_treatment_comparison(
    df,
    df_final
)


# Analyse target variable distribution

plot_target_distribution(
    df_final,
    target_col='future_clv'
)


# Generate pairwise relationship analysis

plot_feature_pairplot(
    df_final,
    output_dir="outputs"
)



# ==============================================================================
# 6. FEATURE SELECTION AND MULTICOLLINEARITY CONTROL (SECTION 3.4.4)
# ==============================================================================

# Separate predictor variables and target variable

X = df_final.drop(
    columns=['id', 'future_clv']
)

y = df_final['future_clv']


# Apply recursive VIF filtering
# to remove highly correlated predictors

X_vif_selected, vif_report, dropped_features = calculate_vif_recursive(
    X,
    thresh=10.0
)


# Save VIF analysis results

vif_report.to_csv(
    "outputs/vif_after_filtering.csv",
    index=False
)


# Restore non-numerical variables after VIF filtering

X_final_selected = X[
    X_vif_selected.columns.tolist()
    +
    X.select_dtypes(exclude=['number']).columns.tolist()
]



# ==============================================================================
# 7. DATASET PARTITIONING (SECTION 3.3.4)
# ==============================================================================

# Split dataset into training, validation and testing subsets
# Training: 70%
# Validation: 20%
# Testing: 10%

X_train, X_temp, y_train, y_temp = train_test_split(
    X_final_selected,
    y,
    test_size=0.30,
    random_state=42
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.3333,
    random_state=42
)



# ==============================================================================
# 8. FEATURE SCALING (SECTION 3.3.3)
# ==============================================================================

# StandardScaler is fitted only using training data
# to prevent information leakage

X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_val,
    X_test
)



# ==============================================================================
# 9. STATISTICAL FEATURE SELECTION ANALYSIS (SECTION 3.4.4)
# ==============================================================================

# Correlation analysis
run_correlation_analysis(
    X_train_scaled,
    y_train
)


# Mutual Information analysis
mi_scores = run_mutual_information(
    X_train_scaled,
    y_train
)



# ==============================================================================
# 10. CUSTOMER SEGMENTATION ANALYSIS (SECTION 3.4.4)
# ==============================================================================

# Convert scaled data back into dataframe format

if not isinstance(X_train_scaled, pd.DataFrame):

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

else:
    X_train_scaled_df = X_train_scaled.copy()


# Select behavioural features for clustering

cluster_features = [
    'engagement_adjusted_value',
    'avg_purchase_value',
    'loyalty_status'
]


available_features = [
    feature for feature in cluster_features
    if feature in X_train_scaled_df.columns
]


# Generate customer segmentation visualization

plot_customer_segments_2d(
    df_scaled=X_train_scaled_df[available_features],
    df_original=df_final.loc[X_train.index],
    output_dir="outputs"
)



# ==============================================================================
# 11. MACHINE LEARNING MODELLING PIPELINE (SECTION 3.6)
# ==============================================================================

# Execute:
# - Model training
# - Hyperparameter tuning
# - Cross-validation
# - Model evaluation
# - SHAP interpretation

run_machine_learning_pipeline(

    X_train_scaled=X_train_scaled,
    X_val_scaled=X_val_scaled,
    X_test_scaled=X_test_scaled,

    y_train=y_train,
    y_val=y_val,
    y_test=y_test,

    features_list=X_final_selected.columns.tolist(),

    mi_scores=mi_scores

)



# ==============================================================================
# 12. BUSINESS VALUE ANALYSIS (SECTION 4.4.1)
# ==============================================================================

# Analyse predicted CLV contribution across customer segments

plot_clv_segment_distributions(
    df_final,
    output_dir="outputs"
)