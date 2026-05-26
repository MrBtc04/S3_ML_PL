import os
import sys
import time

# Ensure src directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_cwru_data, create_windows
from feature_engineering import extract_features, scale_features
from model_isolation_forest import train_isolation_forest
from model_lstm_autoencoder import train_lstm_autoencoder
from model_xgboost import train_xgboost
from evaluate import compare_models

def run_pipeline():
    """
    Orchestrates the entire S3 ML Catenaria Diagnostic & Anomaly Detection Pipeline.
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("S3 ML PIPELINE — STARTING EXECUTION (CATENARIA DICTIONARY)")
    print("="*80)
    
    # Create required folder structure
    for path in ["data/raw", "data/processed", "models/isolation_forest", 
                 "models/lstm_autoencoder", "models/xgboost", "output/reports", "output/plots"]:
        os.makedirs(path, exist_ok=True)
        
    try:
        # Step 1 & 2: Load Catenaria data and align timeseries
        print("\n--- STEP 1 & 2: LOADING & CHRONOLOGICAL TIMESERIES ALIGNMENT ---")
        load_cwru_data(raw_dir="data/raw", processed_dir="data/processed")
        create_windows(raw_dir="data/raw", processed_dir="data/processed")
        
        # Step 3 & 4: Feature extraction and scaling
        print("\n--- STEP 3 & 4: SENSOR FEATURE CLEANING & STANDARD SCALING ---")
        extract_features(windows_path="data/processed/windows.csv", features_path="data/processed/features.csv")
        scale_features(features_path="data/processed/features.csv", 
                       scaled_path="data/processed/features_scaled.csv", 
                       scaler_path="models/scaler.pkl")
        
        # Step 5 & 6: Isolation Forest Anomaly Detection
        print("\n--- STEP 5 & 6: ISOLATION FOREST TRAINING & INFERENCE ---")
        train_isolation_forest(scaled_path="data/processed/features_scaled.csv",
                               model_path="models/isolation_forest/if_model.pkl",
                               plot_dir="output/plots")
                               
        # Step 7 & 8: LSTM Autoencoder Anomaly Detection
        print("\n--- STEP 7 & 8: LSTM AUTOENCODER TRAINING & INFERENCE ---")
        train_lstm_autoencoder(scaled_path="data/processed/features_scaled.csv",
                               model_path="models/lstm_autoencoder/lstm_ae_model.keras",
                               plot_dir="output/plots")
                               
        # Step 9 & 10: XGBoost Fault Classifier
        print("\n--- STEP 9 & 10: XGBOOST ANOMALY CLASSIFIER TRAINING ---")
        train_xgboost(scaled_path="data/processed/features_scaled.csv",
                      model_path="models/xgboost/xgb_model.json",
                      report_path="output/reports/xgboost_report.txt",
                      plot_dir="output/plots")
                      
        # Step 11: Combined comparative evaluation
        print("\n--- STEP 11: COMBINED EVALUATION & AGREEMENT COMPARISON ---")
        compare_models(scaled_path="data/processed/features_scaled.csv",
                       results_path="output/reports/combined_results.csv",
                       plot_path="output/plots/model_comparison.png")
                       
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("\n" + "="*80)
        print("S3 ML CATENARIA PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total Execution Time: {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
        print("="*80 + "\n")
        
        print("Expected Outputs Created:")
        print("  - Models:")
        print("    * models/scaler.pkl")
        print("    * models/isolation_forest/if_model.pkl")
        print("    * models/lstm_autoencoder/lstm_ae_model.keras")
        print("    * models/xgboost/xgb_model.json")
        print("  - Reports:")
        print("    * output/reports/xgboost_report.txt")
        print("    * output/reports/combined_results.csv")
        print("  - Plots:")
        print("    * output/plots/if_score_distribution.png")
        print("    * output/plots/lstm_training_loss.png")
        print("    * output/plots/confusion_matrix.png")
        print("    * output/plots/feature_importance.png")
        print("    * output/plots/model_comparison.png")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        print("="*80 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
