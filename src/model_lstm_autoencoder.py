import os
# Suppress TensorFlow logging to keep console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, time_steps=30):
    """
    Creates overlapping sequences of a specified length (time_steps) from a 2D numpy array.
    Shape output: (num_samples, time_steps, num_features)
    """
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

def train_lstm_autoencoder(scaled_path="data/processed/features_scaled.csv",
                           model_path="models/lstm_autoencoder/lstm_ae_model.keras",
                           plot_dir="output/plots"):
    """
    Prepares sequences, trains the LSTM Autoencoder, calculates reconstruction error,
    flags anomalies based on threshold, and saves the updated DataFrame and plots.
    """
    print(f"[*] Loading scaled features from {scaled_path}...")
    df = pd.read_csv(scaled_path)
    
    # 12 scaled feature columns
    feature_cols = [c for c in df.columns if c not in ["label", "if_prediction", "if_score", "lstm_error", "lstm_anomaly"]]
    X_all_raw = df[feature_cols].values
    y_all = df["label"].values
    
    # 1. Prepare training data (normal data only: label == 0)
    normal_idx = (y_all == 0)
    X_normal_raw = X_all_raw[normal_idx]
    
    time_steps = 30
    print(f"[*] Creating sequences of length {time_steps} for training (normal data only)...")
    X_train = create_sequences(X_normal_raw, time_steps)
    print(f"    - Normal training sequences shape: {X_train.shape}")
    
    if len(X_train) == 0:
        raise ValueError("No enough normal data points to form a sequence of length 30!")
        
    # 2. Build the LSTM Autoencoder model
    print("[*] Building LSTM Autoencoder architecture...")
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(time_steps, X_train.shape[2]), return_sequences=False),
        RepeatVector(time_steps),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X_train.shape[2]))
    ])
    
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    # 3. Train the model
    print("[*] Training LSTM Autoencoder...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=True,
        verbose=1
    )
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[*] LSTM Autoencoder model saved to {model_path}")
    
    # 4. Save training loss plot
    print("[*] Saving training loss plot...")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.plot(history.history['loss'], label='Train Loss', color='#1F77B4', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#FF7F0E', linewidth=2)
    plt.title('LSTM Autoencoder Training & Validation Loss', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss (MAE)', fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    loss_plot_path = os.path.join(plot_dir, "lstm_training_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[*] Training loss plot saved to {loss_plot_path}")
    
    # 5. Run inference on all data
    # Create sequences from the entire dataset
    print("[*] Creating sequences for all data...")
    X_all_seq = create_sequences(X_all_raw, time_steps)
    print(f"    - All sequences shape: {X_all_seq.shape}")
    
    print("[*] Running inference (reconstruction) on all sequences...")
    X_reconstructed = model.predict(X_all_seq)
    
    # Calculate MAE for each sequence
    mae_seq = np.mean(np.abs(X_all_seq - X_reconstructed), axis=(1, 2))
    
    # Calculate reconstruction error on normal training sequences to set the threshold
    print("[*] Calculating anomaly threshold...")
    X_train_reconstructed = model.predict(X_train)
    mae_train = np.mean(np.abs(X_train - X_train_reconstructed), axis=(1, 2))
    
    mean_val = np.mean(mae_train)
    std_val = np.std(mae_train)
    threshold = mean_val + 3 * std_val
    print(f"    - Normal MAE: Mean = {mean_val:.6f}, Std = {std_val:.6f}")
    print(f"    - Anomaly Threshold (Mean + 3*Std) = {threshold:.6f}")
    
    # Detect anomalies
    anomalies_seq = np.where(mae_seq > threshold, "Yes", "No")
    
    # Align sequence results back to original rows
    # The first (time_steps - 1) rows don't have enough history, so we pad them
    lstm_errors = np.zeros(len(df))
    lstm_anomalies = np.array(["No"] * len(df), dtype=object)
    
    # Fill in the sequence results
    # Each sequence index `i` corresponds to row index `i + time_steps - 1`
    lstm_errors[time_steps - 1:] = mae_seq
    lstm_anomalies[time_steps - 1:] = anomalies_seq
    
    df["lstm_error"] = lstm_errors
    df["lstm_anomaly"] = lstm_anomalies
    
    # Save back to features_scaled.csv
    df.to_csv(scaled_path, index=False)
    print(f"[*] Updated features_scaled.csv with 'lstm_error' and 'lstm_anomaly'")
    
    # Quick eval check
    normal_count = np.sum(df["label"] == 0)
    normal_anoms = np.sum((df["label"] == 0) & (df["lstm_anomaly"] == "Yes"))
    fault_count = np.sum(df["label"] > 0)
    fault_anoms = np.sum((df["label"] > 0) & (df["lstm_anomaly"] == "Yes"))
    
    print("\n" + "="*40)
    print("LSTM AUTOENCODER ANOMALY DETECTION RESULTS")
    print("="*40)
    print(f"Normal windows flagged as anomalies (False Positives): {normal_anoms}/{normal_count} ({normal_anoms/normal_count*100:.2f}%)")
    print(f"Faulty windows flagged as anomalies (True Positives): {fault_anoms}/{fault_count} ({fault_anoms/fault_count*100:.2f}%)")
    print("="*40 + "\n")
    
    return scaled_path

if __name__ == "__main__":
    train_lstm_autoencoder()
