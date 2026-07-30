import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

# =============================================================================
# SECTION 3.3 DATA PREPROCESSING
# =============================================================================

# =============================================================================
# Section 3.3.1 - Missing Value Handling and Categorical Encoding
# =============================================================================

def clean_data(df):
    """
    Implements the preprocessing procedures described in 
    Section 3.3.1 (Handle Missing Values) and Section 3.3.3 (Data Transformation).
    """
    # -------------------------------------------------------------------------
    # Step 1: Remove records with excessive missing values
    # Methodology: Section 3.3.1 - Elimination of Records of High Missingness
    # Records containing more than 50% missing values are removed.
    # -------------------------------------------------------------------------
    threshold = len(df.columns) / 2
    df = df.dropna(thresh=threshold).copy()


    # -------------------------------------------------------------------------
    # Step 2: Impute missing numerical values
    # Methodology: Section 3.3.1 - Numerical Imputation
    #
    # - Median imputation for highly skewed variables (skewness > 1)
    # - Mean imputation for approximately symmetric variables
    # -------------------------------------------------------------------------
    numerical_cols = ['age', 'income', 'purchase_amount', 'satisfaction_score']
    for col in numerical_cols:
        if col in df.columns:
            # Fill based on skewness as per Section 3.3.1
            fill_val = df[col].median() if df[col].skew() > 1 else df[col].mean()
            df[col] = df[col].fillna(fill_val)

    # -------------------------------------------------------------------------
    # Step 3: Imputation of Isolated Missing Values (Categorical)
    # Methodology: Section 3.3.1 - Categorical Imputation
    # Mode imputation is applied to preserve the most frequent category.
    # -------------------------------------------------------------------------
    categorical_cols = [
        'gender', 'education', 'region', 
        'loyalty_status','purchase_frequency', 'product_category', 'promotion_usage'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------------------------------------------------
    # Step 4: Outlier Detection and Treatment
    # Methodology: Section 3.3.2
    # -------------------------------------------------------------------------
    df = handle_outliers(df)

    # =============================================================================
    # Section 3.3.3 - Data Transformation
    # =============================================================================

    # -------------------------------------------------------------------------
    # Ordinal Encoding
    #
    # Variables with an inherent ordering are converted into numerical values
    # while preserving their natural ranking.
    # -------------------------------------------------------------------------

    # Purchase Frequency: Rare (1), Occasional (2), Frequent (3)
    if 'purchase_frequency' in df.columns:
        df['purchase_frequency'] = df['purchase_frequency'].astype(str).str.strip().str.lower()
        freq_map = {'rare': 1, 'occasional': 2, 'frequent': 3}
        df['purchase_frequency'] = df['purchase_frequency'].map(freq_map)
        # Fill any gaps with the mode (most common frequency rank) 
        df['purchase_frequency'] = df['purchase_frequency'].fillna(df['purchase_frequency'].mode()[0]).astype(int)

    # Education: High School (0), College (1), Bachelor (2), Masters (3)
    if 'education' in df.columns:
        df['education'] = df['education'].astype(str).str.strip().str.lower()
        edu_map = {'highschool': 0, 'college': 1, 'bachelor': 2, 'masters': 3}
        df['education'] = df['education'].map(edu_map)
        # Fill any gaps with the mode (most common education rank) [cite: 1136]
        df['education'] = df['education'].fillna(df['education'].mode()[0]).astype(int)

    # -------------------------------------------------------------------------
    # Binary Encoding
    #
    # Variables containing two categories are converted into binary values.
    # -------------------------------------------------------------------------
    
    # Gender: Male (0), Female (1)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 0, 'female': 1})
        
    # Promotion Usage: No (0), Yes (1)
    if 'promotion_usage' in df.columns:
        if df['promotion_usage'].dtype == 'object':
            df['promotion_usage'] = df['promotion_usage'].astype(str).str.lower().str.strip().map({'no': 0, 'yes': 1})
        
        # Ensure it is numeric and fill any stray nulls
        df['promotion_usage'] = pd.to_numeric(df['promotion_usage'], errors='coerce').fillna(0).astype(int)
        
    # Loyalty Status: Regular (0), Silver (1), Gold (2)
    if 'loyalty_status' in df.columns:
        status_map = {'regular': 0, 'silver': 1, 'gold': 2} 
        df['loyalty_status'] = df['loyalty_status'].astype(str).str.strip().str.lower().map(status_map).fillna(0)


    # -------------------------------------------------------------------------
    # One-Hot Encoding
    #
    # Nominal variables are transformed into dummy variables.
    # drop_first=True is applied to reduce multicollinearity.
    # -------------------------------------------------------------------------
    nominal_vars = ['region', 'product_category']
    df = pd.get_dummies(df, columns=nominal_vars, drop_first=True, dtype=int)

    return df

# =============================================================================
# Section 3.3.2 - Outlier Detection and Treatment
# =============================================================================
def handle_outliers(df):
    """
    Implements Methodology Section 3.3.2: Outlier Detection and Treatment
    """
    financial_cols = ['income', 'purchase_amount']
    
    for col in financial_cols:
        if col in df.columns:
            # -------------------------------------------------------------
            # Step 1: Detect potential outliers using the IQR method
            # -------------------------------------------------------------
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # -------------------------------------------------------------
            # Step 2: Winsorization
            #
            # Values below the 1st percentile and above the
            # 99th percentile are capped.
            # -------------------------------------------------------------
            df[col] = winsorize(df[col], limits=[0.01, 0.01])
            
            # -------------------------------------------------------------
            # Step 3: Log Transformation
            #
            # log1p() is applied to reduce skewness while safely
            # handling zero values.
            # -------------------------------------------------------------
            df[f'log_{col}'] = np.log1p(df[col])
            
    return df

# =============================================================================
# Section 3.3.3 - Numerical Scaling
# =============================================================================

#Section 3.3.3: Numerical Variable Scaling & Standardization (Data Transformation)
def scale_features(X_train, X_val, X_test, continuous_cols=None):
    """
    Implements Methodology Section 3.3.3: Numerical Variable Scaling & Standardization
    Guarantees numerical stability and prevents disproportionate variable impact.
    """

    # Initialize the StandardScaler as justified in your text
    scaler = StandardScaler()
    
    # Identify all numerical predictors available after preprocessing
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Create copies to avoid rewriting underlying slices incorrectly
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    # Fit the scaler using the training data only to prevent data leakage
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # Apply the fitted scaler to validation and testing datasets
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler