import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

def train_xgboost(scaled_path="data/processed/features_scaled.csv",
                  model_path="models/xgboost/xgb_model.json",
                  report_path="output/reports/xgboost_report.txt",
                  plot_dir="output/plots"):
    """
    Trains an XGBoost binary classifier on physical features, Isolation Forest score,
    and LSTM Autoencoder error. Generates evaluation reports, confusion matrices, and importances.
    """
    print(f"[*] Loading scaled features from {scaled_path}...")
    if not os.path.exists(scaled_path):
        raise FileNotFoundError(f"Scaled features not found at {scaled_path}. Ensure IF and LSTM training have run.")
        
    df = pd.read_csv(scaled_path)
    
    # Feature columns: 5 Catenaria features + 2 anomaly metrics
    feature_cols = ["Altezza", "Taglia", "Temperatura", "Umidita", "Vento", "if_score", "lstm_error"]
    
    # Check that all features exist
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Required feature column '{col}' is missing in the scaled features DataFrame.")
            
    X = df[feature_cols]
    y = df["label"]
    
    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"[*] Splitting dataset:")
    print(f"    - Train shape: {X_train.shape}")
    print(f"    - Test shape: {X_test.shape}")
    print(f"    - Class distribution in train: {np.bincount(y_train)}")
    print(f"    - Class distribution in test: {np.bincount(y_test)}")
    
    # 2. Train XGBoost binary classifier
    print("[*] Training XGBoost Binary Classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"[*] XGBoost model saved to {model_path}")
    
    # 3. Predict on test set for evaluation and predict on all data
    print("[*] Running predictions on test set and entire dataset...")
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X)
    
    # Add predictions back to features_scaled.csv
    df["xgb_prediction"] = y_all_pred
    df.to_csv(scaled_path, index=False)
    print(f"[*] Updated features_scaled.csv with 'xgb_prediction'")
    
    # 4. Evaluate and save reports
    acc = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(
        y_test, y_test_pred, 
        target_names=["Normal (0)", "Anomaly (1)"]
    )
    
    print("\n" + "="*40)
    print("XGBOOST CLASSIFIER TEST METRICS")
    print("="*40)
    print(f"Accuracy: {acc:.6f}")
    print(f"Classification Report:\n{class_report}")
    print("="*40 + "\n")
    
    # Save report to text file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("XGBOOST ANOMALY CLASSIFIER REPORT (CATENARIA)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {acc:.6f}\n\n")
        f.write("Classification Report per Class:\n")
        f.write(class_report + "\n")
        f.write("="*60 + "\n")
    print(f"[*] Detailed classification report saved to {report_path}")
    
    # 5. Plot and save Confusion Matrix
    print("[*] Plotting confusion matrix...")
    os.makedirs(plot_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_test_pred)
    plt.style.use('default')
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Normal", "Anomaly"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    
    # Premium modifications
    ax.grid(False)
    plt.title("XGBoost Anomaly Classifier Confusion Matrix", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    
    cm_plot_path = os.path.join(plot_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix plot saved to {cm_plot_path}")
    
    # 6. Plot and save Feature Importance
    print("[*] Plotting feature importances...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(sorted_features)))
    
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color=colors, edgecolor="none", height=0.6)
    plt.title("XGBoost Catenaria Feature Importance Analysis", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.tight_layout()
    
    fi_plot_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"[*] Feature importance plot saved to {fi_plot_path}")
    
    return scaled_path

if __name__ == "__main__":
    train_xgboost()
