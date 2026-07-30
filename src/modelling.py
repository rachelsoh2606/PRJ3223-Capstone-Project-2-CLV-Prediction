import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.selection import run_permutation_importance
from src.visualization import (plot_predictions_vs_actual, plot_residual_distributions, 
                               plot_cross_validation_stability, plot_global_importance_comparison, plot_shap_dependence)

def run_machine_learning_pipeline(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, features_list, mi_scores):
    """
    SECTION 3.6 MACHINE LEARNING MODELLING & PROCEDURES ---

    Implements:
    - Model selection
    - Model training
    - Hyperparameter optimisation
    - Cross-validation
    - Performance evaluation
    - Explainable AI analysis

    """
    print("\n==================================================")
    print("   INITIALIZING TUNING & CV PIPELINE (SEC 3.6.2)  ")
    print("==================================================")

    # ==============================================================
    # Section 3.6.2: Cross-Validation Strategy
    # ==============================================================

    # Section 3.6.2(b): Define a robust 5-Fold Cross-Validation wrapper strategy.
    # This systematically smooths training variance and drops the risk of model overfitting.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ==============================================================
    # Section 3.6.1: Linear Regression Baseline Model
    # ==============================================================

    # Linear Regression is used as an interpretable baseline model
    # for comparison against advanced ensemble algorithms.

    print("\n[1/3] Training Linear Regression Baseline...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Generate baseline validation split predictions and compute performance metrics
    lr_preds = lr_model.predict(X_val_scaled)

    # Calculate baseline evaluation metrics
    lr_r2 = r2_score(y_val, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_val, lr_preds))
    lr_mae = mean_absolute_error(y_val, lr_preds)


    # ==============================================================
    # Section 3.6.1 & 3.6.2: Random Forest Model
    # Model Selection and Hyperparameter Optimisation
    # ==============================================================

    # Random Forest is selected as an ensemble learning model
    # to capture non-linear relationships between customer features
    # and future CLV.

    print("\n[2/3] Optimizing Random Forest Regressor via RandomizedSearchCV...")

    # Define hyperparameter search space
    rf_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize Random Forest estimator
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Apply RandomizedSearchCV with five-fold validation
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=10,  # Dynamically samples 10 independent hyperparameter configurations
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Train and identify the best hyperparameter combination
    rf_search.fit(X_train_scaled, y_train)
    best_rf_model = rf_search.best_estimator_
    print(f"Optimal Random Forest Hyperparameters: {rf_search.best_params_}")


    # Pulls the negative MSE scores for the best parameter index, converts to positive, and square-roots to get RMSE
    best_index_rf = rf_search.best_index_
    rf_folds_rmse = [np.sqrt(-rf_search.cv_results_[f'split{i}_test_score'][best_index_rf]) for i in range(5)]
    print(f"Random Forest CV Fold RMSE Scores: {rf_folds_rmse}")

    # Generate validation predictions using the optimized hyperparameter settings
    rf_preds = best_rf_model.predict(X_val_scaled)

    # Calculate Random Forest evaluation metrics
    rf_r2 = r2_score(y_val, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
    rf_mae = mean_absolute_error(y_val, rf_preds)


   # ==============================================================
    # Section 3.6.1 & 3.6.2: XGBoost Model
    # Advanced Gradient Boosting Optimisation
    # ==============================================================

    # XGBoost is implemented as the final advanced model because
    # of its ability to capture complex non-linear patterns.
    print("\n[3/3] Optimizing XGBoost Regressor via RandomizedSearchCV...")

    # Set up parameter variations focusing on learning rate, subsample density, estimators, and tree depth
    xgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    # Initialise XGBoost regression model
    xgb_base = XGBRegressor(random_state=42, n_jobs=-1)

    # Perform hyperparameter optimisation
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=10,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Train model using optimised parameters
    xgb_search.fit(X_train_scaled, y_train)

    best_xgb_model = xgb_search.best_estimator_
    print(f"Optimal XGBoost Hyperparameters: {xgb_search.best_params_}")

    #  EXTRACT THE 5 FOLDS FOR XGBOOST
    best_index_xgb = xgb_search.best_index_
    xgb_folds_rmse = [np.sqrt(-xgb_search.cv_results_[f'split{i}_test_score'][best_index_xgb]) for i in range(5)]
    print(f"XGBoost CV Fold RMSE Scores: {xgb_folds_rmse}")

    # Generate validation predictions 
    xgb_preds = best_xgb_model.predict(X_val_scaled)

    # Calculate XGBoost performance metrics
    xgb_r2 = r2_score(y_val, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
    xgb_mae = mean_absolute_error(y_val, xgb_preds)


    # ==============================================================
    # Section 3.6.3: Model Evaluation and Performance Comparison
    # ==============================================================

    # Generate a summary table to compare model performance
    # using R², MAE, and RMSE evaluation metrics.

    print("\n==================================================")
    print("   PHASE 1: MODEL SELECTION & VALIDATION METRICS   ")
    print("==================================================")
    
    results_df_val = pd.DataFrame({
        'Model': ['Linear Regression (Baseline)', 'Optimized Random Forest', 'Optimized XGBoost'],
        'R-squared (R2)': [lr_r2, rf_r2, xgb_r2],
        'MAE': [lr_mae, rf_mae, xgb_mae],
        'RMSE': [lr_rmse, rf_rmse, xgb_rmse]
    })
    print(results_df_val.to_string(index=False))

    # ==============================================================
    # Section 3.6.3: Unseen Test Set Evaluation
    # ==============================================================

    # Final evaluation is performed on an unseen test dataset
    # to assess model generalisation capability.
    
    test_lr_preds = lr_model.predict(X_test_scaled)
    test_rf_preds = best_rf_model.predict(X_test_scaled)
    test_xgb_preds = best_xgb_model.predict(X_test_scaled)

    print("\n==================================================")
    print("   PHASE 2: NEW FINAL UNSEEN TEST SUMMARY TABLE   ")
    print("==================================================")
    
    results_df_test = pd.DataFrame({
        'Model (Unseen Test Set)': ['Linear Regression (Baseline)', 'Optimized Random Forest', 'Optimized XGBoost'],
        'R-squared (R2)': [r2_score(y_test, test_lr_preds), r2_score(y_test, test_rf_preds), r2_score(y_test, test_xgb_preds)],
        'MAE': [mean_absolute_error(y_test, test_lr_preds), mean_absolute_error(y_test, test_rf_preds), mean_absolute_error(y_test, test_xgb_preds)],
        'RMSE': [np.sqrt(mean_squared_error(y_test, test_lr_preds)), np.sqrt(mean_squared_error(y_test, test_rf_preds)), np.sqrt(mean_squared_error(y_test, test_xgb_preds))]
    })
    print(results_df_test.to_string(index=False))
    
    # Export final test performance table
    results_df_test.to_csv("outputs/final_test_performance_summary.csv", index=False)
    
    # Automatically saves performance tracking table to outputs file directory
    results_df_val.to_csv("outputs/model_performance_summary.csv", index=False)
    print("\nUpdated performance summary configurations saved to outputs/model_performance_summary.csv")

    # ==============================================================
    # Section 3.6.3: Prediction Error Visualisation
    # ==============================================================

    # Generate visual diagnostics to compare predicted CLV
    # against actual future CLV values.

    plot_predictions_vs_actual(y_val, lr_preds, rf_preds, xgb_preds, output_dir="outputs")
    print(f"Model prediction vs actual scatter plot successfully saved to outputs/model_predictions_vs_actual.png")

    # Residual analysis evaluates prediction error distribution
    # and identifies potential model bias.
    plot_residual_distributions(y_val, lr_preds, rf_preds, xgb_preds, output_dir="outputs")

    # ==============================================================
    # Section 3.6.2: Cross-Validation Stability Analysis
    # ==============================================================

    # Evaluate model robustness by comparing RMSE variation
    # across different validation folds.
    plot_cross_validation_stability(rf_folds_rmse, xgb_folds_rmse, output_dir="outputs")

    # ==============================================================================
    # --- NEW ADDITION: MANAGERIAL DEPLOYMENT - TOP 10 FUTURE CLV FORECASTS ---
    # ==============================================================================
    # We construct a strategic tracking array using predictions from our top-performing model (XGBoost)
    # This maps back to the validation indices to extract direct actionable targets for retention teams.
    print("\n[Managerial Operations] Extracting Top 10 High-Value Future CLV Targets...")
    
    # Compile a clean customer-level prediction tracking table
    customer_predictions = pd.DataFrame({
        'Customer_Index': y_val.index,
        'Actual_Future_CLV': y_val.values,
        'Predicted_Future_CLV': xgb_preds
    })
    
    # Sort descending based on our model's forecasts to isolate high-value segments
    top_10_clv_table = customer_predictions.sort_values(by='Predicted_Future_CLV', ascending=False).head(10).reset_index(drop=True)
    
    # Round metrics cleanly for executive business presentation styles
    top_10_clv_table['Actual_Future_CLV'] = top_10_clv_table['Actual_Future_CLV'].round(2)
    top_10_clv_table['Predicted_Future_CLV'] = top_10_clv_table['Predicted_Future_CLV'].round(2)
    
    # Save the finalized operational table to disk for easy thesis reproduction
    top_10_clv_table.to_csv("outputs/top_10_future_clv_targets.csv", index=False)
    print("\n--- TOP 10 STRATEGIC CUSTOMER FORECASTS (XGBOOST ENGINE) ---")
    print(top_10_clv_table.to_string())
    print("\nOperational target list saved to: outputs/top_10_future_clv_targets.csv")

    # ==============================================================
    # Section 3.6.4(a): Permutation Feature Importance
    # ==============================================================

    # Permutation importance evaluates feature contribution by
    # measuring the decrease in model performance after shuffling
    # individual features

    print("\n[Model-Based Selection] Executing Permutation Feature Importance Plots...")
    run_permutation_importance(best_rf_model, X_val_scaled, y_val, "Random Forest")
    run_permutation_importance(best_xgb_model, X_val_scaled, y_val, "XGBoost")
    print("Permutation plots generated and saved inside outputs/ folder.")


    # ==============================================================
    # Section 3.6.4: Explainable Artificial Intelligence (XAI)
    # ==============================================================

    print("\n==================================================")
    print("   INITIALIZING XAI INTERPRETABILITY (SEC 3.6.4)  ")
    print("==================================================")

    # ==============================================================
    # Section 3.6.4(b): Built-in XGBoost Feature Importance
    # ==============================================================

    # Extract feature importance scores directly from the trained
    # XGBoost model to identify influential predictors.

    print("[1/2] Extracting Built-in Feature Importance from XGBoost...")

    # Get feature importance scores from the trained model
    xgb_importances = best_xgb_model.feature_importances_

    # Organize into a clean DataFrame for interpretation
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Gini_Importance': xgb_importances
    }).sort_values(by='Gini_Importance', ascending=False)

    print("\nTop 5 Built-in Predictors (XGBoost):")
    print(importance_df.head(5).to_string(index=False))

    # Save the built-in importance ranking table to disk
    importance_df.to_csv("outputs/xgboost_builtin_importance.csv", index=False)


    # ==============================================================
    # Section 3.6.4(c): SHAP Global Interpretation
    # ==============================================================

    # SHAP explains individual prediction contributions by estimating
    # how each feature influences the final model output.

    print("\n[2/2] Computing SHAP Values for Model Explanations...")

    # Initialize the SHAP TreeExplainer, specifically optimized for tree ensemble models
    explainer = shap.TreeExplainer(best_xgb_model)

    # Compute SHAP values on the scaled validation set to assess feature trajectories
    shap_values = explainer(X_val_scaled)

    # Standardize formats to extract the underlying raw matrix safely
    shap_matrix = shap_values.values if hasattr(shap_values, 'values') else shap_values

    # ==============================================================================
    # --- ADDED: COMPUTE AND PRINT GLOBAL SHAP FEATURE IMPACT IN TEXT ---
    # ==============================================================================
    print("\n=====================================================================")
    print("[XAI Diagnostics] Global SHAP Feature Importance Rankings:")
    print("=====================================================================")
    
    # Calculate the mean absolute SHAP value for each column across all observations
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    
    # Bundle into a clean Pandas DataFrame for programmatic sorting
    global_shap_df = pd.DataFrame({
        'Feature Name': X_val_scaled.columns,
        'Mean Absolute SHAP (Impact Magnitude)': mean_abs_shap
    })
    
    # Sort so the highest-impact variables sit securely at the top
    global_shap_df = global_shap_df.sort_values(by='Mean Absolute SHAP (Impact Magnitude)', ascending=False)
    
    # Print the plain text results to the console output
    print(global_shap_df.to_string(index=False, formatters={'Mean Absolute SHAP (Impact Magnitude)': '{:,.4f}'.format}))
    print("=====================================================================\n")
    # ==============================================================================

    # Configure a fresh plot frame for the SHAP summary (Beeswarm) plot
    plt.figure(figsize=(10, 6))

    # Generate the global interpretability summary plot
    shap.summary_plot(shap_values, X_val_scaled, show=False)
    plt.title("SHAP Global Feature Contribution Summary", fontsize=14, pad=15)

    # Save high-resolution graphic for thesis document integration
    plt.savefig("outputs/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------------
    # --- GENERATE INDIVIDUAL SHAP DEPENDENCE PLOTS & TEXT LOGS ---
    # ----------------------------------------------------------------------
    print("\n[EDA Diagnostics] Initializing Dynamic XAI Feature Dependence Audit...")

    # Let Pandas dynamically read the actual features that survived feature selection
    active_columns = X_val_scaled.columns.tolist()
    
    # We select the top 3 dominant features present in your model to display in the text output
    # This guarantees that whatever columns generated your images will appear in the text!
    xai_features_to_inspect = [col for col in ['loyalty_status', 'purchase_frequency', 'purchse_frequency', 'satisfaction_score', 'engagement_adjusted_value', 'avg_purchase_value'] if col in active_columns]
    
    # Fallback backup safety check
    if len(xai_features_to_inspect) == 0:
        xai_features_to_inspect = active_columns[:3]


    # ==============================================================
    # Section 3.6.4(d): SHAP Dependence Analysis
    # ==============================================================

    # SHAP dependence plots are generated to examine the relationship
    # between important features and predicted CLV outcomes.

    plot_shap_dependence(shap_values, X_val_scaled, xai_features_to_inspect, output_dir="outputs/shap_dependence")
    
    # EXTRACT TREES IMPORTANCE AND EXECUTE THE GLOBAL METRIC COMPARISON 
    print("\n[Consensus Diagnostics] Extracting structural trees frameworks importance metrics...")
    
    # ==============================================================
    # Section 3.6.4(e): Multi-method Feature Importance Comparison
    # ==============================================================

    # Compare feature importance obtained from:
    # - Mutual Information
    # - Random Forest
    # - XGBoost
    
    rf_importances = best_rf_model.feature_importances_
    xgb_importances = best_xgb_model.feature_importances_
    
    # Call our cross-methodology comparison plot
    plot_global_importance_comparison(features_list, mi_scores, rf_importances, xgb_importances, output_dir="outputs")


    # ---------------------------------------------------------
    # NEW: Export Customer-Level Predictions (Validation Set)
    # ---------------------------------------------------------
    print("\n[Exporting Predictions] Compiling final customer-level prediction metrics...")
    
    # Create a DataFrame containing ground truths and predictions side-by-side
    predictions_df = pd.DataFrame({
        'Actual_CLV': y_val.values if hasattr(y_val, 'values') else y_val,
        'LR_Predicted_CLV': lr_preds,
        'RF_Predicted_CLV': rf_preds,
        'XGBoost_Predicted_CLV': xgb_preds
    }, index=y_val.index) # Keeps the original customer index/ID references if present
    
    # Save the individual predictions to a CSV file
    predictions_df.to_csv("outputs/customer_clv_predictions_val.csv", index=True)
    print("Customer-level predicted CLV values successfully saved to outputs/customer_clv_predictions_val.csv")

    print("XAI Evaluation complete. Visualizations saved to outputs/shap_summary_plot.png")