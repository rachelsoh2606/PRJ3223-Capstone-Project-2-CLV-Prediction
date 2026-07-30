import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

# =============================================================================
# Section 3.4.4 - Correlation Analysis
# =============================================================================
def run_correlation_analysis(X, y):
    """
    Methodology Section 3.4.4: Correlation Analysis
    Investigates linear (Pearson) and monotonic (Spearman) relationships.

    """
    # Combine predictors and target variable for correlation analysis
    data = X.copy()
    data['future_clv'] = y
    
    # -------------------------------------------------------------------------
    # Pearson Correlation Analysis
    #
    # Measures the linear relationship between numerical features and CLV.
    # Correlation values range from -1 to +1.
    # -------------------------------------------------------------------------
    correlation_matrix = data.corr(
        method='pearson'
    )

    # Generate correlation heatmap visualization
    plt.figure(figsize=(14, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot_kws={"size": 8}
    )

    plt.title("Pearson Correlation Heatmap")

    plt.savefig(
        "outputs/correlation_heatmap.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    
    # -------------------------------------------------------------------------
    # Spearman Correlation Analysis
    #
    # Measures monotonic relationships, which are suitable for ordinal
    # variables and non-linear trends.
    # -------------------------------------------------------------------------
    spearman_corr = data.corr(method='spearman')['future_clv'].sort_values(ascending=False)
    
    # Save ranking results for reporting
    spearman_corr.to_csv("outputs/spearman_rankings.csv")
    print("\nSpearman Rankings saved to outputs/spearman_rankings.csv")

    return spearman_corr

# =============================================================================
# Section 3.4.4 - Variance Inflation Factor (VIF)
# =============================================================================

def calculate_vif_recursive(X_frame, thresh=10.0):
    """
    Methodology Section 3.4.4: VIF Analysis.

    Feature with high multicollinearity (VIF > 10) are iteratively removed to ensure model stability.
    """

    # Select numerical variables for VIF calculation
    X_numeric = X_frame.select_dtypes(include=['number']).copy()
    
    # Remove constant variables to avoid calculation errors
    X_numeric = X_numeric.loc[:, X_numeric.var() > 0]
    
    dropped_features = []
    loop_count = 1
    
    # -------------------------------------------------------------------------
    # Recursive VIF elimination
    #
    # The feature with the highest VIF is removed iteratively until the
    # remaining predictors have acceptable multicollinearity levels.
    # -------------------------------------------------------------------------
    while True:
        columns = X_numeric.columns.tolist()
        if len(columns) <= 1:
            break
            
        vif_values = []

        # Calculate VIF values across the remaining numeric data frame vectors
        for i in range(len(columns)):
            try:
                vif = variance_inflation_factor(X_numeric.values, i)
                vif_values.append(vif)
            except Exception:
                vif_values.append(np.inf)
                
        vif_df = pd.DataFrame({'Feature': columns, 'VIF': vif_values})
        vif_df = vif_df.sort_values(by='VIF', ascending=False).reset_index(drop=True)
        
        max_vif = vif_df.iloc[0]['VIF']
        max_feature = vif_df.iloc[0]['Feature']
        
        # Stop when all variables satisfy VIF threshold
        if max_vif <= thresh or np.isnan(max_vif):
            break
            
        # Log the dynamic removal details to the console log
        print(f"  [VIF Iteration {loop_count}] Dropping '{max_feature}' with unstable VIF = {max_vif:.2f}")

        X_numeric = X_numeric.drop(columns=[max_feature])
        dropped_features.append(max_feature)
        loop_count += 1
        
    # Generate final VIF report
    final_vif_report = pd.DataFrame({
        'Feature': X_numeric.columns,
        'VIF': [
            variance_inflation_factor(
                X_numeric.values,
                idx
            )
            for idx in range(X_numeric.shape[1])
        ]
    })

    final_vif_report = final_vif_report.sort_values(
        by='VIF',
        ascending=False
    )
    
    return X_numeric, final_vif_report, dropped_features

# =============================================================================
# Section 3.4.4 - Mutual Information Analysis
# =============================================================================

def run_mutual_information(X, y):
    """
    Methodology Section 3.4.4: Mutual Information Analysis
    
    Identifies non-linear dependencies between features and the target variable.
    """

    print("\n[Selection Diagnostics] Computing Mutual Information scores...")
    
    # Select numerical features for MI calculation
    X_numeric = X.select_dtypes(include=['number']).copy()
    
    # Calculate MI scores
    mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
    
    # Convert results into ranked dataframe
    mi_series = pd.Series(mi_scores, index=X_numeric.columns).sort_values(ascending=False)
    
    # Visualize feature dependency scores
    plt.figure(figsize=(10, 6))

    sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, legend=False, palette="viridis")
    plt.title("Data-Driven Feature Dependency: Mutual Information Scores", fontsize=12, weight='bold', pad=10)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Customer Attributes")
    plt.tight_layout()
    
    # Save chart asset to outputs
    plt.savefig("outputs/mutual_information_plot.png", dpi=300, bbox_inches='tight')

    mi_series.to_csv("outputs/mutual_information_scores.csv", header=["Mutual_Information_Score"]) #
    
    plt.close()
    
    print("Mutual Information scores calculated and saved to: outputs/mutual_information_plot.png")

    return mi_series

# =============================================================================
# Section 3.6.3 - Permutation Feature Importance
# =============================================================================

def run_permutation_importance(model, X_val, y_val, model_name):
    """
    Methodology Section 3.6.3: Permutation Feature Importance

    Measures the direct drop in model performance when feature vectors are shuffled.
    """
    print(f"Calculating Permutation Feature Importance for {model_name}...")
    
    # Calculate permutation importance scores
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Rank features based on importance scores
    sorted_importances_idx = result.importances_mean.argsort()[::-1]
    features = np.array(X_val.columns)[sorted_importances_idx]
    importances = result.importances_mean[sorted_importances_idx]
    
   # Save feature importance results
    perm_df = pd.DataFrame({'Feature': features, 'Permutation_Importance_Mean': importances})
    perm_df.to_csv(f"outputs/{model_name.lower().replace(' ', '_')}_permutation_importance.csv", index=False)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, hue=features, legend=False, palette="mako")
    plt.title(f"Model-Based Importance: {model_name} Permutation Metric Profile", fontsize=12, weight='bold', pad=10)
    plt.xlabel("Decrease in Model Accuracy score")
    plt.tight_layout()
    
    plt.savefig(f"outputs/{model_name.lower().replace(' ', '_')}_permutation_plot.png", dpi=300, bbox_inches='tight')
    plt.close()