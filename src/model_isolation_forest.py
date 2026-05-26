import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def train_isolation_forest(scaled_path="data/processed/features_scaled.csv",
                           model_path="models/isolation_forest/if_model.pkl",
                           plot_dir="output/plots"):
    """
    Trains an Isolation Forest model on scaled normal features, runs inference on
    all data, saves the predictions, and creates evaluation plots.
    """
    print(f"[*] Loading scaled features from {scaled_path}...")
    df = pd.read_csv(scaled_path)
    
    # Separate features and label
    y = df["label"].values
    X = df.drop(columns=["label"])
    
    # 1. Train on normal data only (label == 0)
    normal_idx = (y == 0)
    X_train = X[normal_idx]
    
    print(f"[*] Training Isolation Forest on {len(X_train)} normal samples...")
    model = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    model.fit(X_train)
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[*] Isolation Forest model saved to {model_path}")
    
    # 2. Run inference on all data
    print("[*] Running inference on all data...")
    if_predictions = model.predict(X)          # 1 = normal, -1 = anomaly
    if_scores = model.decision_function(X)       # More negative = more anomalous
    
    # Add columns to the DataFrame
    df["if_prediction"] = if_predictions
    df["if_score"] = if_scores
    
    # Save back to scaled features
    df.to_csv(scaled_path, index=False)
    print(f"[*] Updated scaled features with 'if_prediction' and 'if_score' in {scaled_path}")
    
    # 3. Evaluate
    normal_anomalies = np.sum((y == 0) & (if_predictions == -1))
    normal_total = np.sum(y == 0)
    fault_anomalies = np.sum((y > 0) & (if_predictions == -1))
    fault_total = np.sum(y > 0)
    
    print("\n" + "="*40)
    print("ISOLATION FOREST EVALUATION RESULTS")
    print("="*40)
    print(f"Normal windows flagged as anomalies (False Positives): {normal_anomalies}/{normal_total} ({normal_anomalies/normal_total*100:.2f}%)")
    print(f"Faulty windows flagged as anomalies (True Positives): {fault_anomalies}/{fault_total} ({fault_anomalies/fault_total*100:.2f}%)")
    print("="*40 + "\n")
    
    # 4. Plot anomaly score distribution
    print("[*] Generating Isolation Forest anomaly score plot...")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plt.hist(if_scores[y == 0], bins=50, alpha=0.6, label="Normal Data", color="#008080", edgecolor="none")
    plt.hist(if_scores[y > 0], bins=50, alpha=0.6, label="Faulty Data", color="#D9534F", edgecolor="none")
    
    plt.axvline(x=0.0, color="black", linestyle="--", linewidth=1.5, label="Anomaly Threshold (0.0)")
    plt.title("Isolation Forest Anomaly Score Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Decision Function Anomaly Score (Higher = More Normal)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, "if_score_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[*] Anomaly score distribution plot saved to {plot_path}")
    
    return scaled_path

if __name__ == "__main__":
    train_isolation_forest()
