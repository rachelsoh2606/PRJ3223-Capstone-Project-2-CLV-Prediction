import os
import pandas as pd

def check_file_status(folder_path, filename, registered_set):
    """Checks file existence and updates the registered tracking set."""
    full_path = os.path.join(folder_path, filename)
    normalized_path = os.path.normpath(full_path)
    registered_set.add(normalized_path)
    
    if os.path.exists(full_path):
        return f"[FOUND] {filename}"
    else:
        return f"[MISSING / SKIPPED] {filename}"

def read_csv_to_text(folder_path, filename, registered_set, rows_limit=None):
    """Safely reads a tabular CSV dataset and logs its tracking registration path."""
    full_path = os.path.join(folder_path, filename)
    normalized_path = os.path.normpath(full_path)
    registered_set.add(normalized_path)
    
    if not os.path.exists(full_path):
        return f"File '{filename}' was not generated or was pruned by selection matrix logic.\n"
    
    try:
        df = pd.read_csv(full_path)
        if rows_limit:
            return df.head(rows_limit).to_string(index=False)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading {filename}: {str(e)}\n"

def build_pipeline_manifest(outputs_dir="outputs"):
    """
    Scrapes generated data files, tracks visual image assets, 
    and adds a definitive safety scan for unregistered artifacts.
    """
    manifest_path = os.path.join(outputs_dir, "pipeline_results_manifest.txt")
    print(f"Scraping analysis folder data maps inside: {outputs_dir}...")
    
    # Track every single file path explicitly checked by our script logic
    registered_files = set()
    # Ensure the manifest output path itself is ignored during safety checks
    registered_files.add(os.path.normpath(manifest_path))
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("=====================================================================\n")
        f.write("               PIPELINE EXECUTION RAW MANIFEST MATRIX                \n")
        f.write("=====================================================================\n\n")
        
        # -------------------------------------------------------------------
        # TRACKING SECTION 1: GLOBAL ARTIFACT STATUS MAP (PNG & CSV ASSETS)
        # -------------------------------------------------------------------
        f.write("--- SECTION 1: GRAPHICS AND METRIC FILES AUDIT MAP ---\n")
        
        # A. Exploratory Data Analysis Distribution Visual Assets
        dist_dir = os.path.join(outputs_dir, "distributions")
        f.write("\n[A. EDA & Distribution Profiles Asset Map]\n")
        f.write(f"  - {check_file_status(dist_dir, 'age_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'income_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'purchase_amount_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'satisfaction_score_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'purchase_frequency_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'log_income_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'log_purchase_amount_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'gender_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'education_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'region_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'loyalty_status_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'product_category_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'income_treatment_comparison.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'purchase_amount_treatment_comparison.png', registered_files)}\n")
        f.write(f"  - {check_file_status(dist_dir, 'future_clv_target_distribution.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'feature_relationship_pairplot.png', registered_files)}\n")
        
        # B. Feature Pruning & Clustering Architecture Visual Assets
        f.write("\n[B. Variable Audit, Multicollinearity, and Cluster Segments Map]\n")
        f.write(f"  - {check_file_status(outputs_dir, 'correlation_heatmap.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'mutual_information_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'customer_segments_pca.png', registered_files)}\n")
        
        # C. Modeling Accuracy and Validation Visual Assets
        f.write("\n[C. Regressions and Ensemble Performance Diagnostics Map]\n")
        f.write(f"  - {check_file_status(outputs_dir, 'model_predictions_vs_actual.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'model_residual_distributions.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'model_cv_stability_boxplot.png', registered_files)}\n")
        
        # D. Explainable AI (XAI) Local Interpretation Visual Assets
        shap_dir = os.path.join(outputs_dir, "shap_dependence")
        f.write("\n[D. Explainable AI Trajectories Matrix Map]\n")
        f.write(f"  - {check_file_status(outputs_dir, 'shap_summary_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(shap_dir, 'loyalty_status_dependence_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(shap_dir, 'satisfaction_score_dependence_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(shap_dir, 'log_income_dependence_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'feature_importance_global_comparison.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'random_forest_permutation_plot.png', registered_files)}\n")
        f.write(f"  - {check_file_status(outputs_dir, 'xgboost_permutation_plot.png', registered_files)}\n")
        f.write("\n" + "="*69 + "\n\n")
        
        # -------------------------------------------------------------------
        # TRACKING SECTION 2: EXTRACT NUMERICAL TABLES (CSV SCRAPING BLOCKS)
        # -------------------------------------------------------------------
        f.write("--- SECTION 2: DATA REGISTRY LOGS AND COMPUTED ARRAYS ---\n\n")
        
        f.write("--- LOG 2.A: QUANTITATIVE SUMMARY COMPILATION MODULE (CHAPTER 4) ---\n")
        f.write(read_csv_to_text(outputs_dir, "model_performance_summary.csv", registered_files))
        f.write("\n\n")
        
        f.write("--- LOG 2.B: RECURSIVE VARIABLE INTERDEPENDENCY REPORT (VIF STATUS) ---\n")
        f.write(read_csv_to_text(outputs_dir, "vif_after_filtering.csv", registered_files))
        f.write("\n\n")
        
        f.write("--- LOG 2.C: MULTI-LEVEL ALGORITHMIC PREDICTOR GAIN VECTOR (XGBOOST) ---\n")
        f.write(read_csv_to_text(outputs_dir, "xgboost_builtin_importance.csv", registered_files, rows_limit=10))
        f.write("\n\n")
        
        f.write("--- LOG 2.D: RANDOM FOREST PERMUTATION SHUFFLE ACCURACY DECREMENT VECTORS ---\n")
        f.write(read_csv_to_text(outputs_dir, "random_forest_permutation_importance.csv", registered_files))
        f.write("\n\n")
        
        f.write("--- LOG 2.E: XGBOOST PERMUTATION SHUFFLE ACCURACY DECREMENT VECTORS ---\n")
        f.write(read_csv_to_text(outputs_dir, "xgboost_permutation_importance.csv", registered_files))
        f.write("\n\n")
        
        f.write("--- LOG 2.F: EXECUTIVE STRATEGIC RETENTION BLUEPRINT (TOP 10 FORECASTS) ---\n")
        f.write(read_csv_to_text(outputs_dir, "top_10_future_clv_targets.csv", registered_files))
        f.write("\n\n")
        
        # -------------------------------------------------------------------
        # TRACKING SECTION 3: AUTOMATED DIRECTORY SCANNER Check (CRITICAL SAFETY)
        # -------------------------------------------------------------------
        f.write("="*69 + "\n")
        f.write("--- SECTION 3: SYSTEM INTEGRITY DIRECTORY INTEGRITY SCAN (CRITICAL SAFETY) ---\n")
        f.write("="*69 + "\n\n")
        
        untracked_assets = []
        # Walk through your directories to pick up any files not registered above
        for root, _, files in os.walk(outputs_dir):
            for file in files:
                current_file_path = os.path.normpath(os.path.join(root, file))
                if current_file_path not in registered_files:
                    # Capture the file relative to the output folder path cleanly
                    relative_display_path = os.path.relpath(current_file_path, outputs_dir)
                    untracked_assets.append(relative_display_path)
                    
        if len(untracked_assets) > 0:
            f.write(f"[WARNING] Detected {len(untracked_assets)} unlisted/historical files on disk:\n")
            for asset in sorted(untracked_assets):
                f.write(f"  -> outputs/{asset}\n")
            f.write("\nNote: These elements are physical outputs that were not checked by our code logic script details.\n")
        else:
            f.write("[SAFE] Clean scan complete. 100% of the physical files match your script metrics list.\n")

    print(f"[Success] File tracking scrape completed. Clean text report saved directly to: {manifest_path}")
if __name__ == "__main__":
    build_pipeline_manifest()