import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# ==============================================================================
# SECTION 3.3.2: OUTLIER DETECTION AND TREATMENT
# SECTION 3.4.3: EXPLORATORY DATA ANALYSIS
# ==============================================================================

def plot_feature_distributions(df, columns_to_plot, output_dir="outputs/distributions"):
    """
    Methodology Section 3.3.2 & 3.4.3: Feature Distribution Analysis

    Generate numerical feature distibutions plots.
    Used to examine skewness, distributed patterns and potential outliers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set standard academic plotting style
    sns.set_theme(style="whitegrid")
    
    for col in columns_to_plot:
        if col not in df.columns:
           continue
            
        # Create histogram with KDE and boxplot for distribution analysis
        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, 
            figsize=(10, 6), 
            sharex=True, 
            gridspec_kw={"height_ratios": (.8, .2)}
        )
        
        # 1. Plot the Histogram + Kernel Density Estimate (KDE)
        sns.histplot(data=df, x=col, kde=True, ax=ax_hist, color="steelblue", bins=30)
        ax_hist.set_title(f"Distribution Profile Analysis: {col.replace('_', ' ').title()}", fontsize=12, pad=10)
        ax_hist.set_ylabel("Frequency/Density")
        ax_hist.set_xlabel("") # Remove X label from top plot to prevent overlapping
        
        # 2. Plot the Boxplot to cleanly highlight upper/lower outliers
        sns.boxplot(data=df, x=col, ax=ax_box, color="lightcoral", flierprops={"markerfacecolor": "red", "markersize": 5})
        ax_box.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        
        # Clean up layout structure
        plt.tight_layout()
        
        # Save high-resolution copy for thesis document
        file_name = f"{output_dir}/{col}_distribution.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Distribution plots successfully generated and saved inside: {output_dir}/")

# ==============================================================================
# SECTION 3.3.3: DATA TRANSFORMATION
# ==============================================================================

def plot_categorical_distributions(df, columns_to_plot, output_dir="outputs/distributions"):
    """
    Methodology Section 3.3.3: Categorical Variable Distribution Analysis
    
    Generates categorical variable frequency distributions.
    Used to examine cateory balance before encoding. 
    """
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    for col in columns_to_plot:
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(8, 5))
        
        # Count plots displays category frequency
        ax = sns.countplot(data=df, x=col, hue=col, legend=False, palette="muted", order=df[col].value_counts().index)
        
        plt.title(f"Frequency Distribution: {col.replace('_', ' ').title()}", fontsize=12, pad=10)
        plt.ylabel("Customer Count")
        plt.xlabel(col.replace('_', ' ').title())
        plt.xticks(rotation=45 if df[col].nunique() > 3 else 0) # Rotate text if many categories
        
        # Add text count labels on top of the bars for thesis clarity
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 3),
                        textcoords='offset points')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Categorical plots successfully saved inside: {output_dir}/")

# ==============================================================================
# SECTION 3.3.2: OUTLIER DETECTION AND TREATMENT
# ==============================================================================

def plot_outlier_treatment_comparison(df_raw, df_engineered, output_dir="outputs/distributions"):
    """
    Methodology Section 3.3.2: Outlier Treatment Comparison Analysis

    Compares raw and transformed financial features.
    Used to demonstrate the effectiveness of winsorization and log transformation

    """
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Define the pairs to compare: (Raw column name, Engineered column name)
    comparison_pairs = [
        ('income', 'log_income'),
        ('purchase_amount', 'log_purchase_amount')
    ]
    
    for raw_col, eng_col in comparison_pairs:
        # Verify both attributes are available across data structures
        if raw_col not in df_raw.columns or eng_col not in df_engineered.columns:
            continue
            
        # Initialize a 2x2 grid framework for a comprehensive structural comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
        
        # ------------------------------
        # Raw feature distribution
        # ------------------------------
        sns.histplot(data=df_raw, x=raw_col, kde=True, ax=axes[0, 0], color="crimson", bins=30)
        axes[0, 0].set_title(f"BEFORE: Raw {raw_col.replace('_', ' ').title()} Distribution", fontsize=11, weight='bold')
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_xlabel("")
        
        # Bottom Left: Raw Boxplot exposing heavy outlier tails
        sns.boxplot(data=df_raw, x=raw_col, ax=axes[1, 0], color="lightcoral", 
                    flierprops={"markerfacecolor": "red", "markersize": 4})
        axes[1, 0].set_xlabel(f"Raw Values ({raw_col.replace('_', ' ').title()})")
        
        # ------------------------------
        # Transformed feature distribution
        # ------------------------------
        sns.histplot(data=df_engineered, x=eng_col, kde=True, ax=axes[0, 1], color="forestgreen", bins=30)
        axes[0, 1].set_title(f"AFTER: Log-Transformed & Winsorized Profile", fontsize=11, weight='bold')
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_xlabel("")
        
        # Bottom Right: Transformed Boxplot showing eliminated outliers
        sns.boxplot(data=df_engineered, x=eng_col, ax=axes[1, 1], color="lightgreen", 
                    flierprops={"markerfacecolor": "darkgreen", "markersize": 4})
        axes[1, 1].set_xlabel(f"Treated Values ({eng_col})")
        
        # Polish title parameters and spacing
        plt.suptitle(f"Methodological Impact Analysis: Outlier Mitigation ({raw_col.upper()})", 
                     fontsize=14, weight='bold', y=0.98)
        plt.tight_layout()
        
        # Save high-resolution copy for the document
        file_path = f"{output_dir}/{raw_col}_treatment_comparison.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Outlier treatment comparison plots saved to: {output_dir}/")

# ==============================================================================
# SECTION 3.5: TARGET VARIABLE ANALYSIS
# ==============================================================================

def plot_target_distribution(df, target_col='future_clv', output_dir="outputs/distributions"):
    """
    Methodology Section 3.5: Target Variable (Future CLV) Structural Analysis

    Generates the distribution of the computed CLV target.
    Used to examine target skewness and value distribution.

    """

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return
        
    # Set up a clean, high-clarity 2-panel layout sharing the horizontal scale
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, 
        figsize=(10, 6), 
        sharex=True, 
        gridspec_kw={"height_ratios": (.8, .2)}
    )
    
    # 1. Top Panel: Histogram and Kernel Density Estimate curve
    sns.histplot(data=df, x=target_col, kde=True, ax=ax_hist, color="darkorchid", bins=40)
    ax_hist.set_title(f"Target Variable Profile Analysis: Computed Customer Lifetime Value (CLV)", fontsize=13, weight='bold', pad=12)
    ax_hist.set_ylabel("Customer Count / Density")
    ax_hist.set_xlabel("")
    
    # 2. Bottom Panel: Boxplot outlining the full concentration spread and extreme value outliers
    sns.boxplot(data=df, x=target_col, ax=ax_box, color="plum", flierprops={"markerfacecolor": "purple", "markersize": 4})
    ax_box.set_xlabel("Historical Customer Lifetime Value (Currency Units)", fontsize=11, weight='bold')
    
    # Adjust spacing dynamically
    plt.tight_layout()
    
    # Save the high-resolution graphic for thesis chapter integration
    file_path = f"{output_dir}/{target_col}_target_distribution.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Target variable distribution plot saved directly to: {file_path}")


# ==============================================================================
# SECTION 3.4.4: CUSTOMER SEGMENTATION ANALYSIS
# ==============================================================================

def plot_customer_segments_2d(df_scaled, df_original, output_dir="outputs"):
    """
    Methodology Section 3.4.4: Unsupervised Customer Segmentation Matrix
    
    Performs K-means clustering and PCA projection.
    Used to visualize customer segments in 2D.

    """

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="white")
    
    # 1. Select numerical features
    clustering_vars = df_scaled.select_dtypes(include=['number']).columns.tolist()
    if len(clustering_vars) < 2:
        print("Warning: Insufficient numeric dimensions remaining for clustering. Skipping.")
        return
        
    X_cluster = df_scaled[clustering_vars]
    
    # 2. Fit K-Means (perform customer-clustering)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # 3. Apply PCA (reduce dimensions for visualization)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cluster)
    
    plot_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    plot_df['Cluster'] = cluster_labels
    
    # Map the generated cluster labels back to their matching training row indices
    cluster_series = pd.Series(cluster_labels, index=df_scaled.index, name='Cluster')
    
    # Re-slice original scale data frame using the identical training rows index vector
    analysis_df = df_original.loc[df_scaled.index].copy()
    analysis_df['Cluster'] = cluster_series
    
    # Let Pandas dynamically collect ONLY the meaningful numeric columns present in the source dataframe
    available_raw_metrics = [
        col for col in ['engagement_adjusted_value', 'avg_purchase_value', 'loyalty_status', 'satisfaction_score', 'age'] 
        if col in analysis_df.columns
    ]
    
    # Compute clean averages grouping by the verified Cluster column
    cluster_means = analysis_df.groupby('Cluster')[available_raw_metrics].mean()
    
    # Sort dynamically based on who generated the highest engagement value footprint
    sort_col = 'engagement_adjusted_value' if 'engagement_adjusted_value' in cluster_means.columns else available_raw_metrics[0]
    sorted_clusters = cluster_means[sort_col].sort_values(ascending=False).index.tolist()
    
    # Reorder so Premium High-Value Core always displays as row 1
    cluster_means = cluster_means.reindex(sorted_clusters)
    cluster_means.index = ["Premium High-Value Core", "Mid-Tier Occasional", "Low-Value / Inactive Tier"]
    
    print("\n=====================================================================")
    print("[Segmentation Diagnostics] Cluster Profile Real-World Averages:")
    print("=====================================================================")
    print(cluster_means.round(2).to_string())
    print("=====================================================================\n")
    
    # Map Cluster integers to descriptive business personas based on data trends
    persona_mapping = {
        sorted_clusters[0]: "Premium High-Value Core",
        sorted_clusters[1]: "Mid-Tier Occasional",
        sorted_clusters[2]: "Low-Value / Inactive Tier"
    }
    plot_df['Strategic Segment'] = plot_df['Cluster'].map(persona_mapping)
    
    # Draw the PCA 2D Cluster Dispersion Space
    plt.figure(figsize=(11, 7))
    segment_colors = {"Premium High-Value Core": "darkblue", 
                      "Mid-Tier Occasional": "darkorange", 
                      "Low-Value / Inactive Tier": "crimson"}
    
    sns.scatterplot(
        data=plot_df, 
        x='Principal Component 1', 
        y='Principal Component 2', 
        hue='Strategic Segment',
        palette=segment_colors,
        alpha=0.8,
        s=60,
        edgecolor='w',
        linewidth=0.5
    )
    
    var_exp = pca.explained_variance_ratio_
    plt.title(f"Unsupervised Customer Segmentation: PCA 2D Coordinate Projection", fontsize=13, weight='bold', pad=12)
    plt.xlabel(f"Principal Component 1 (Explains {var_exp[0]*100:.1f}% Variance)", fontsize=10)
    plt.ylabel(f"Principal Component 2 (Explains {var_exp[1]*100:.1f}% Variance)", fontsize=10)
    plt.legend(title="Strategic Customer Segments", loc="best", frameon=True, shadow=False)
    plt.tight_layout()
    
    file_path = f"{output_dir}/customer_segments_pca.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Customer segment clustering plot generated and saved to: {file_path}")


# ==============================================================================
# SECTION 3.4.4: FEATURE RELATIONSHIP ANALYSIS
# ==============================================================================

def plot_feature_pairplot(df, output_dir="outputs"):
    """
    Methodology Section 3.4.4: Pairwise Feature Relationship Matrix

    Generates pairwise relationships between key features.
    Used to identify interaction patterns among variables.

    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a clean, professional white background style suitable for academia
    sns.set_theme(style="ticks")
    
    # Define the exact key variables requested
    key_variables = ['income', 'purchase_amount', 'purchase_frequency', 'satisfaction_score', 'future_clv']
    
    # Verify columns exist before filtering the dataframe
    available_cols = [col for col in key_variables if col in df.columns]
    plot_df = df[available_cols].copy()
    
    # Group customers by CLV level
    plot_df['Value Tier'] = pd.qcut(plot_df['future_clv'], q=3, labels=['Bronze (Low CLV)', 'Silver (Mid CLV)', 'Gold (High CLV)'])
    
    print("\n[EDA Diagnostics] Generating publication-quality pairwise relationship matrix pairplot...")
    
    # Generate pairwise relationship matrix
    g = sns.pairplot(
        plot_df, 
        vars=available_cols,
        hue='Value Tier',
        palette='viridis',
        kind='scatter',
        diag_kind='kde',
        height=3.0,          
        aspect=1.1,
        plot_kws={'alpha': 0.6, 's': 25, 'edgecolor': 'none'},
        diag_kws={'fill': True, 'alpha': 0.5}
    )
    # Seaborn hides inner labels by default. We loop through the axes array grid
    # to explicitly restore X and Y descriptions across all grid coordinates.
    for i, row_var in enumerate(available_cols):
        for j, col_var in enumerate(available_cols):
            ax = g.axes[i, j]
            if ax is not None:
                # Format strings to clean up database underscores for the thesis presentation
                clean_row_name = row_var.replace('_', ' ').title()
                clean_col_name = col_var.replace('_', ' ').title()
                
                # Assign labels to every internal grid subplot block explicitly
                ax.set_ylabel(clean_row_name, fontsize=9)
                ax.set_xlabel(clean_col_name, fontsize=9)
                
                # Keep tick marks visible inside the grid blocks
                ax.tick_params(axis='both', which='both', labelleft=True, labelbottom=True, labelsize=7)

    # Refine titles and spacing so they match professional textbook aesthetics
    g.fig.suptitle("Pairwise Structural Association Matrix & Target Stratification", y=1.03, fontsize=16, weight='bold')
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1.0), title="Strategic Customer Segments")
    
    plt.tight_layout()
    # Save the high-resolution visualization asset
    file_path = f"{output_dir}/feature_relationship_pairplot.png"
    g.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pairplot feature matrix generated and saved to: {file_path}")

# ==============================================================================
# SECTION 3.7: MODEL EVALUATION
# ==============================================================================

def plot_predictions_vs_actual(y_actual, lr_preds, rf_preds, xgb_preds, output_dir="outputs"):
    """
    Methodology Section 3.7: Model Evaluation and Performance Analysis

    Generates predicted vs. actual scatter plots for multiple models.
    Used to visually assess model accuracy, bias, and variance.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    

    print("\n=====================================================================")
    print("[Model Validation Diagnostics] Empirical Prediction Evaluation (Chapter 4):")
    print("=====================================================================")
    
    cv_perf_list = []
    models_to_eval = [
        ('Linear Regression (Baseline)', lr_preds),
        ('Optimized Random Forest', rf_preds),
        ('Optimized XGBoost', xgb_preds)
    ]
    
    for name, preds in models_to_eval:
        r2 = r2_score(y_actual, preds)
        mae = mean_absolute_error(y_actual, preds)
        rmse = root_mean_squared_error(y_actual, preds)
        
        cv_perf_list.append({
            "Model Pipeline": name,
            "R-squared (R2)": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        
    perf_df = pd.DataFrame(cv_perf_list)
    print(perf_df.to_string(index=False))
    print("=====================================================================\n")
    # ==============================================================================
    # Configure a 3-panel layout frame for side-by-side model auditing
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    
    models_data = [
        ('Linear Regression (Baseline)', lr_preds, axes[0], 'royalblue'),
        ('Optimized Random Forest', rf_preds, axes[1], 'darkorange'),
        ('Optimized XGBoost', xgb_preds, axes[2], 'forestgreen')
    ]
    
    # Dynamically find the absolute data boundaries to make sure the diagonal identity line is perfect
    all_values = np.concatenate([y_actual, lr_preds, rf_preds, xgb_preds])
    min_val, max_val = all_values.min(), all_values.max()
    
    for model_name, preds, ax, color in models_data:
        # Draw the customer prediction coordinate points
        ax.scatter(y_actual, preds, alpha=0.4, s=20, color=color, edgecolor='none')
        
        # Draw the standard reference Identity Line (y = x) representing a 100% accurate model
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        
        # Set clean titles and labels matching professional report structures
        ax.set_title(model_name, fontsize=12, weight='bold', pad=10)
        ax.set_xlabel("Actual CLV (Currency)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc='upper left', fontsize=9)
    
    axes[0].set_ylabel("Predicted CLV (Currency)", fontsize=11, weight='bold')
    
    plt.suptitle("Model Evaluation Matrix: Predicted vs. Actual Customer Lifetime Value (CLV)", 
                 fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    
    # Export high-resolution chart asset for thesis integration
    file_path = f"{output_dir}/model_predictions_vs_actual.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model prediction vs actual scatter plot successfully saved to: {file_path}")

# ==============================================================================
# SECTION 3.7: RESIDUAL ANALYSIS
# ==============================================================================

def plot_residual_distributions(y_actual, lr_preds, rf_preds, xgb_preds, output_dir="outputs"):
    """
    Methodology Section 3.7: Residual Diagnostic Analysis

    Generates residual distributions for model diagnostics.
    Used to evaluate prediction error patterns.
    """

    from scipy.stats import skew, kurtosis
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Calculate residuals (Actual - Predicted) for every model
    lr_residuals = y_actual - lr_preds
    rf_residuals = y_actual - rf_preds
    xgb_residuals = y_actual - xgb_preds

 
    print("\n=====================================================================")
    print("[Residual Diagnostics] Error Curve Distribution Shapes (Chapter 4):")
    print("=====================================================================")
    
    residual_stats_list = []
    models_residuals_eval = [
        ('Linear Regression (Baseline)', lr_residuals),
        ('Optimized Random Forest', rf_residuals),
        ('Optimized XGBoost', xgb_residuals)
    ]
    
    for name, res in models_residuals_eval:
        residual_stats_list.append({
            "Model Pipeline": name,
            "Mean Error (Bias)": np.mean(res),
            "Std Dev (Spread)": np.std(res),
            "Skewness": skew(res),
            "Kurtosis (Peak)": kurtosis(res),
            "Min Error": np.min(res),
            "Max Error": np.max(res)
        })
        
    res_df = pd.DataFrame(residual_stats_list)
    print(res_df.to_string(index=False))
    print("=====================================================================\n")
    # ==============================================================================
    
    # Configure a 3-panel layout frame for side-by-side model auditing
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_residuals = [
        ('Linear Regression (Baseline)', lr_residuals, axes[0], 'royalblue'),
        ('Optimized Random Forest', rf_residuals, axes[1], 'darkorange'),
        ('Optimized XGBoost', xgb_residuals, axes[2], 'forestgreen')
    ]
    
    for model_name, residuals, ax, color in models_residuals:
        # Render the Error Histogram coupled with a clean Kernel Density Estimate curve
        # Stat="density" ensures curves are scaled natively as smooth continuous probabilities
        sns.histplot(residuals, kde=True, ax=ax, color=color, bins=40, alpha=0.6, stat="density")
        
        # Draw a bold reference line at exactly 0.0 (The ultimate goal of zero-error forecasting)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Threshold')
        
        # Labeling structures
        ax.set_title(model_name, fontsize=12, weight='bold', pad=10)
        ax.set_xlabel("Residual Error Value (Currency)", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        
    
        if "Linear Regression" in model_name:
            # Let the baseline stretch wider to show its poor performance
            ax.set_xlim(-150000, 150000) 
        else:
            # Zoom in perfectly on the high-density peak of the tree-based models
            ax.set_xlim(-40000, 40000)

    # Since axes are not shared, we add labels individually to look clean
    for i in range(3):
        axes[i].set_ylabel("Probability Density", fontsize=11, weight='bold')
    
    plt.suptitle("Model Diagnostic Matrix: Residual Error Distributions (Actual - Predicted)", 
                 fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    
    # Export graphic asset
    file_path = f"{output_dir}/model_residual_distributions.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual error distribution plot successfully saved to: {file_path}")


# ==============================================================================
# SECTION 3.7: CROSS-VALIDATION ANALYSIS
# ==============================================================================

def plot_cross_validation_stability(rf_folds, xgb_folds, output_dir="outputs"):
    """
    Methodology Section 3.7: Cross-Validation Stability Analysis

    Generates cross-validation performance plots.
    Used to evaluate model stability across validation folds.

    """

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Consolidate the extracted fold vectors into a structured DataFrame
    cv_data = pd.DataFrame({
        'Optimized Random Forest': rf_folds,
        'Optimized XGBoost': xgb_folds
    })


    print("\n=====================================================================")
    print("[Model Stability Diagnostics] 5-Fold Cross-Validation RMSE Summary Tables:")
    print("=====================================================================")
    
    # Calculate boxplot summary thresholds dynamically
    stats_list = []
    for model_name in cv_data.columns:
        scores = cv_data[model_name].values
        stats_list.append({
            "Model Architecture": model_name,
            "Minimum (Bottom Whisker)": np.min(scores),
            "25th Percentile (Q1)": np.percentile(scores, 25),
            "Median (Center Line)": np.median(scores),
            "75th Percentile (Q3)": np.percentile(scores, 75),
            "Maximum (Top Whisker)": np.max(scores),
            "Fold Variance (Spread)": np.var(scores)
        })
        
    cv_stats_df = pd.DataFrame(stats_list)
    print(cv_stats_df.round(4).to_string(index=False))
    print("=====================================================================\n")
    # ==============================================================================

    # Melt the DataFrame into a long format suitable for advanced Seaborn indexing
    cv_melted = cv_data.melt(var_name='Model Architecture', value_name='Fold RMSE (Lower is Better)')
    
    plt.figure(figsize=(9, 6))
    
    # Draw the comparison boxplot mapping stability variance
    sns.boxplot(
        data=cv_melted, 
        x='Model Architecture', 
        y='Fold RMSE (Lower is Better)', 
        hue='Model Architecture',             
        legend=False,
        palette=['darkorange', 'forestgreen'],
        width=0.4,
        linewidth=2
    )
    
    # Overlay individual fold data points to show the raw distribution details
    sns.stripplot(
        data=cv_melted, 
        x='Model Architecture', 
        y='Fold RMSE (Lower is Better)', 
        color='black', 
        size=6, 
        jitter=0.05,
        alpha=0.7
    )
    
    plt.title("Cross-Validation Generalization Audit: 5-Fold RMSE Stability", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Evaluated Machine Learning Models", fontsize=11)
    plt.ylabel("Fold Root Mean Squared Error (RMSE)", fontsize=11)
    
    plt.tight_layout()
    
    # Save high-resolution copy for thesis document integration
    file_path = f"{output_dir}/model_cv_stability_boxplot.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-validation stability boxplot successfully saved to: {file_path}")

# ==============================================================================
# SECTION 3.6.4: EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI)
# ==============================================================================

def plot_shap_dependence(shap_values, X_data, features_to_plot=None, output_dir="outputs/shap_dependence"):
    """
    Methodology Section 3.6.4: SHAP Feature Dependence Analysis

    Generates SHAP dependence plots to explain feature impact.
    Used to analyse how important features influence CLV prediction.
    """

    from scipy.stats import spearmanr
    
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Extract SHAP matrix values
    shap_matrix = shap_values.values if hasattr(shap_values, 'values') else shap_values
    
    # Dynamically scan X_data to match whatever generated your graphs
    active_columns = X_data.columns.tolist()
    
    # Prioritize features that are actively present in the dataframe columns
    target_features = ['engagement_adjusted_value', 'avg_purchase_value', 'loyalty_status']
    features_to_run = [f for f in target_features if f in active_columns]
    
    # Fallback backup: use top features if the hardcoded list fails to intersect
    if not features_to_run:
        features_to_run = active_columns[:3]
        
    print("\n=====================================================================")
    print("[XAI Diagnostics] Individual SHAP Feature Dependence Metrics:")
    print("=====================================================================")
    
    xai_dependence_list = []
    
    # Select important features for interpretation
    for feature in features_to_run:
        feature_idx = X_data.columns.get_loc(feature)
        raw_values = X_data[feature].values
        feature_shap = shap_matrix[:, feature_idx]
        
        # 1. Compute Main Effect Monotonicity using Spearman's Rank Correlation
        corr, p_val = spearmanr(raw_values, feature_shap)
        
        # 2. Identify top interacting feature dynamically across the active space
        max_inter_corr = -1
        top_inter_feature = "None Detected"
        for potential_interactor in X_data.columns:
            if potential_interactor == feature:
                continue
            inter_corr = abs(spearmanr(X_data[potential_interactor].values, feature_shap)[0])
            if inter_corr > max_inter_corr:
                max_inter_corr = inter_corr
                top_inter_feature = potential_interactor
        
        xai_dependence_list.append({
            "Evaluated Feature": feature,
            "Spearman Rho (Direction)": corr,
            "P-Value": p_val,
            "Top Interacting Feature": top_inter_feature,
            "Interaction Strength Index": max_inter_corr
        })
        
        
        plt.figure(figsize=(8, 5))

         # Generate SHAP dependence relationship
        shap.dependence_plot(feature, shap_matrix, X_data, show=False)
        plt.title(f"SHAP Dependence Profile: {feature.replace('_', ' ').title()}", fontsize=12, weight='bold', pad=10)
        plt.tight_layout()
        
        file_path = f"{output_dir}/{feature}_dependence_plot.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Print out all calculated rows at once in a unified terminal table block
    res_df = pd.DataFrame(xai_dependence_list)
    print(res_df.round(5).to_string(index=False))
    print("=====================================================================\n")

# ==============================================================================
# SECTION 3.4.4: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

def plot_global_importance_comparison(features_list, mi_scores, rf_importances, xgb_importances, output_dir="outputs"):
    """
    Methodology Section 3.4.4: Global Multi-Method Feature Importance Comparison

    Compares feature importance obtained from different techniques.
    Used to identify consistent predictors influencing CLV.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Helper lambda function to normalize values between 0 and 1 for a fair grouped comparison
    normalize = lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else x
    
    # Construct a structured DataFrame matching features to normalized metrics
    comp_df = pd.DataFrame({
        'Feature': features_list,
        'Mutual Information': normalize(mi_scores),
        'Random Forest (Gini)': normalize(rf_importances),
        'XGBoost (Gain)': normalize(xgb_importances)
    })
    

    print("\n=====================================================================")
    print("[Feature Diagnostics] Global Cross-Methodology Feature Importance Rank:")
    print("=====================================================================")
    
    # Compute the arithmetic mean across all three distinct evaluation engines
    comp_df['Consensus Score'] = comp_df[['Mutual Information', 'Random Forest (Gini)', 'XGBoost (Gain)']].mean(axis=1)
    
    # Sort by consensus score so the absolute most influential predictors sit at the top
    text_summary_df = comp_df.sort_values(by='Consensus Score', ascending=False)
    
    print(text_summary_df.to_string(index=False, formatters={
        'Mutual Information': '{:,.4f}'.format,
        'Random Forest (Gini)': '{:,.4f}'.format,
        'XGBoost (Gain)': '{:,.4f}'.format,
        'Consensus Score': '{:,.4f}'.format
    }))
    print("=====================================================================\n")
    # ==============================================================================

    # Melt the data into long-form format suitable for Seaborn grouped bar charts
    melted_df = comp_df.melt(id_vars='Feature', var_name='Evaluation Method', value_name='Normalized Importance Score')
    
    # Sort the feature listings based on the average importance across all methods to keep the graph clean
    feature_order = comp_df.set_index('Feature').mean(axis=1).sort_values(ascending=False).index.tolist()
    
    plt.figure(figsize=(12, 6))
    
    # Render the grouped bar chart
    sns.barplot(
        data=melted_df,
        x='Feature',
        y='Normalized Importance Score',
        hue='Evaluation Method',
        palette=['royalblue', 'darkorange', 'forestgreen','crimson'],
        order=feature_order
    )
    
    
    plt.title("Cross-Methodology Consensus Matrix: Normalized Feature Importance Comparison", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Evaluated Input Customer Attributes", fontsize=11)
    plt.ylabel("Normalized Significance Metric (Scale: 0.0 - 1.0)", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Analytics Frameworks", loc="upper right")
    
    plt.tight_layout()
    
    # Export high-resolution asset
    file_path = f"{output_dir}/feature_importance_global_comparison.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Global feature importance comparison plot saved directly to: {file_path}")

# ==============================================================================
# SECTION 4.4.1: BUSINESS INTERPRETATION OF CLV PREDICTION
# ==============================================================================

def plot_clv_segment_distributions(df_final, output_dir="outputs"):
    """
    Methodology Section 4.4.1: Business Value of CLV Prediction (RQ3.1)

    Visualizes predicted CLV distributions across customer segments.
    Used to translate model output into business insights.
    """

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Reconstruct the precise operational data slice matching your clustering rows
    analysis_df = df_final.copy()
    
    # 2. Assign customers to tiers dynamically based on their actual business metrics sequence
    conditions = [
        (analysis_df['engagement_adjusted_value'] >= 20000),
        (analysis_df['engagement_adjusted_value'] >= 10000) & (analysis_df['engagement_adjusted_value'] < 20000),
        (analysis_df['engagement_adjusted_value'] < 10000)
    ]
    choices = ["Premium High-Value Core", "Mid-Tier Occasional", "Low-Value / Inactive Tier"]
    analysis_df['Strategic Segment'] = np.select(conditions, choices, default="Low-Value / Inactive Tier")

    # ==============================================================================
    # --- ADDED: PRINT RAW BUSINESS PERFORMANCE METRICS IN TEXT FORMAT ---
    # ==============================================================================
    print("\n=====================================================================")
    print("[Strategic Business Metrics] CLV Distribution Across Customer Segments:")
    print("=====================================================================")
    
    # Calculate key descriptive stats for each corporate value segment
    segment_stats = analysis_df.groupby('Strategic Segment')['future_clv'].agg([
        ('Customer Count', 'count'),
        ('Minimum CLV', 'min'),
        ('Average CLV', 'mean'),
        ('Median CLV', 'median'),
        ('Maximum CLV', 'max'),
        ('Total Strategic Value (Sum)', 'sum')
    ]).reindex(choices) # Keep rows sorted from Premium down to Low-Value
    
    # Calculate proportion contribution of total portfolio value
    total_portfolio_clv = analysis_df['future_clv'].sum()
    segment_stats['Portfolio Value Share (%)'] = (segment_stats['Total Strategic Value (Sum)'] / total_portfolio_clv) * 100
    
    # Print clean text tables to terminal output
    print(segment_stats.round(2).to_string())
    print("=====================================================================\n")
    # ==============================================================================

    # 3. Initialize the visualization layout
    plt.figure(figsize=(12, 6))
    
    segment_colors = {
        "Premium High-Value Core": "darkblue", 
        "Mid-Tier Occasional": "darkorange", 
        "Low-Value / Inactive Tier": "crimson"
    }

    # 4. Render overlaying probability density curves for each strategic tier
    for segment_name, color in segment_colors.items():
        subset = analysis_df[analysis_df['Strategic Segment'] == segment_name]
        sns.kdeplot(
            data=subset['future_clv'],
            fill=True,
            color=color,
            label=segment_name,
            alpha=0.4,
            linewidth=2.5
        )

    # 5. Add academic-level presentation structures
    plt.title("Strategic Asset Valuation: Predicted CLV Distribution Across Customer Segments", 
              fontsize=13, weight='bold', pad=15)
    plt.xlabel("Forecasted Customer Lifetime Value (Future CLV - Currency)", fontsize=11)
    plt.ylabel("Probability Density Distribution", fontsize=11)
    
    # Formatting ticks cleanly
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.legend(title="Strategic Customer Segments", loc="upper right", frameon=True)
    plt.tight_layout()

    # Export fresh asset for Section 4.4.1
    file_path = f"{output_dir}/clv_distribution_by_segments.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Strategic Value Asset] Segmented CLV distribution plot successfully generated at: {file_path}")