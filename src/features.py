import numpy as np
import pandas as pd

# =============================================================================
# SECTION 3.4 FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Methodology Section 3.4 Feature Engineering and Section 3.5 Target Variable Construction.

    This function creates derived features and constructs the future CLV target variable based on transactional and engagement metrics 
    and remove variables that directly leak information about the target to ensure model integrity.

    """
    df_feat = df.copy()
    
    # =============================================================================
    # Section 3.5 - Target Variable Construction
    # =============================================================================

    # -------------------------------------------------------------------------
    # Step 1: Construct Future CLV Target Variable
    #
    # Future CLV is calculated using purchase behaviour and engagement factors:
    #
    # Future CLV =
    # (Purchase Amount × Purchase Frequency)
    # × (1 + Loyalty Effect)
    # × (1 + Promotion Effect)
    #
    # Loyalty and promotion multipliers represent the influence of customer
    # engagement on potential future value.
    # -------------------------------------------------------------------------
    df_feat['future_clv'] = (
        df_feat['purchase_amount'] * df_feat['purchase_frequency'] * (1 + df_feat['loyalty_status'] * 0.05) * (1 + df_feat['promotion_usage'] * 0.03)
    )
    
    # =============================================================================
    # Section 3.4.3 - Derived Features
    # =============================================================================

    # -------------------------------------------------------------------------
    # Step 2: Log Transformation of Financial Variables
    #
    # Log transformation reduces skewness and stabilizes the variance of
    # financial variables while handling zero values using log1p().
    # -------------------------------------------------------------------------
    df_feat['log_purchase_amount'] = np.log1p(df_feat['purchase_amount'])
    df_feat['log_income'] = np.log1p(df_feat['income'])
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Average Purchase Value
    #
    # Represents customer spending intensity per purchase.
    # A small constant is added to prevent division-by-zero errors.
    # -------------------------------------------------------------------------
    df_feat['avg_purchase_value'] = df_feat['purchase_amount'] / (df_feat['purchase_frequency'] + 1e-5)
    
    # -------------------------------------------------------------------------
    # Step 4: Generate Engagement-Adjusted Monetary Value
    #
    # Combines monetary contribution with customer loyalty engagement.
    # Higher loyalty levels increase the estimated customer value.
    # -------------------------------------------------------------------------
    df_feat['engagement_adjusted_value'] = df_feat['purchase_amount'] * (df_feat['loyalty_status'] + 1)
    
    # =============================================================================
    # Section 3.4.5 - Feature Justification
    # Section 3.5.3 - Ensuring Clarity for Model Prediction
    # =============================================================================

    # -------------------------------------------------------------------------
    # Step 5: Remove Target Leakage Variables
    #
    # Purchase amount and purchase frequency are removed because they are
    # directly used in Future CLV calculation. Removing these variables ensures
    # that models learn customer characteristics rather than replicating the
    # target formula.
    #
    # Income is removed because it was considered during target construction
    # exploration and excluded from the final predictive feature set.
    # -------------------------------------------------------------------------
    leakage_cols = ['purchase_amount', 'purchase_frequency', 'income']
    df_feat = df_feat.drop(columns=[col for col in leakage_cols if col in df_feat.columns])
    
    return df_feat  