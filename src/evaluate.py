import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_models(scaled_path="data/processed/features_scaled.csv",
                   results_path="output/reports/combined_results.csv",
                   plot_path="output/plots/model_comparison.png"):
    """
    Combines the results from all three models into a unified report.
    Calculates fault detection rates and exports a comparative bar chart.
    """
    print(f"[*] Loading scaled features with predictions from {scaled_path}...")
    df = pd.read_csv(scaled_path)
    
    # 1. Build combined results table
    # Columns required: Window, True Label, IF Anomaly, LSTM Anomaly, XGBoost Prediction
    combined_df = pd.DataFrame({
        "Window": np.arange(1, len(df) + 1),
        "True Label": df["label"],
        "IF Anomaly": df["if_prediction"],
        "LSTM Anomaly": df["lstm_anomaly"],
        "XGBoost Prediction": df["xgb_prediction"]
    })
    
    # Save combined results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    combined_df.to_csv(results_path, index=False)
    print(f"[*] Combined results saved to {results_path} (Shape: {combined_df.shape})")
    
    # 2. Calculate detection rate per model
    # Only evaluate on fault windows (True Label > 0)
    fault_mask = (df["label"] > 0)
    total_fault_windows = np.sum(fault_mask)
    
    if total_fault_windows == 0:
        print("[!] Warning: No fault windows found in the dataset! Cannot calculate detection rates.")
        return results_path
        
    # Isolation Forest: % of fault windows flagged as anomalies (-1)
    if_detected = np.sum(fault_mask & (df["if_prediction"] == -1))
    if_rate = (if_detected / total_fault_windows) * 100
    
    # LSTM Autoencoder: % of fault windows flagged as anomalies ("Yes")
    lstm_detected = np.sum(fault_mask & (df["lstm_anomaly"] == "Yes"))
    lstm_rate = (lstm_detected / total_fault_windows) * 100
    
    # XGBoost: % of fault windows correctly classified (xgb_prediction == label)
    xgb_correct = np.sum(fault_mask & (df["xgb_prediction"] == df["label"]))
    xgb_rate = (xgb_correct / total_fault_windows) * 100
    
    print("\n" + "="*50)
    print("COMPARATIVE MODEL DIAGNOSTIC RATES (ON FAULT WINDOWS)")
    print("="*50)
    print(f"Total Fault Windows Evaluated: {total_fault_windows}")
    print(f"Isolation Forest Detection Rate:      {if_detected}/{total_fault_windows} ({if_rate:.2f}%)")
    print(f"LSTM Autoencoder Detection Rate:      {lstm_detected}/{total_fault_windows} ({lstm_rate:.2f}%)")
    print(f"XGBoost Classifier Classification Rate: {xgb_correct}/{total_fault_windows} ({xgb_rate:.2f}%)")
    print("="*50 + "\n")
    
    # 3. Plot detection rate per model
    print("[*] Plotting model comparison bar chart...")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    models = ["Isolation Forest\n(Anomaly Detection)", "LSTM Autoencoder\n(Unsupervised)", "XGBoost\n(Supervised Classifier)"]
    rates = [if_rate, lstm_rate, xgb_rate]
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = ["#17A2B8", "#6F42C1", "#28A745"] # Cyan, Purple, Green
    bars = plt.bar(models, rates, color=colors, edgecolor="none", width=0.55)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333"
        )
        
    plt.title("Comparative Performance Diagnostic Rates on CWRU Fault Windows", fontsize=13, fontweight="bold", pad=20)
    plt.ylabel("Diagnostic Accuracy / Detection Rate (%)", fontsize=11)
    plt.ylim(0, 110) # Give extra space on top for values labels
    plt.grid(axis='x', linestyle='') # remove vertical gridlines
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[*] Comparative bar chart saved to {plot_path}")
    
    return results_path

if __name__ == "__main__":
    compare_models()
