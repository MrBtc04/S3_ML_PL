import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def extract_features(windows_path="data/processed/windows.csv", features_path="data/processed/features.csv"):
    """
    Loads the windows from windows.csv, extracts time and frequency domain features
    for each window, and saves the feature matrix.
    """
    print(f"[*] Loading windows from {windows_path}...")
    df = pd.read_csv(windows_path)
    
    label_col = df["label"].values
    # Feature columns are everything except label
    feature_cols = [c for c in df.columns if c != "label"]
    window_data = df[feature_cols].values
    
    num_windows = len(df)
    print(f"[*] Extracting features for {num_windows} windows...")
    
    extracted_features = []
    
    for i in tqdm(range(num_windows), desc="Features"):
        window = window_data[i]
        label = label_col[i]
        
        # Sampling rate logic: CWRU normal is 48k, faults are 12k
        fs = 48000 if label == 0 else 12000
        
        # 1. Time domain features
        mean_val = np.mean(window)
        std_val = np.std(window)
        rms_val = np.sqrt(np.mean(window**2))
        peak_val = np.max(np.abs(window))
        crest_factor = peak_val / rms_val if rms_val > 1e-8 else 0.0
        kurt_val = kurtosis(window)
        skew_val = skew(window)
        p2p_val = np.max(window) - np.min(window)
        
        # 2. Frequency domain features
        fft_vals = np.fft.fft(window)
        n = len(window)
        # First half of spectrum (positive frequencies)
        half_n = n // 2
        mag = np.abs(fft_vals)[:half_n]
        
        fft_mean = np.mean(mag)
        fft_std = np.std(mag)
        fft_peak = np.max(mag)
        
        # Exclude DC component (index 0) for dominant frequency to get a real signal peak
        if len(mag) > 1:
            dominant_idx = np.argmax(mag[1:]) + 1
            dominant_freq = dominant_idx * (fs / n)
        else:
            dominant_freq = 0.0
            
        feature_row = {
            "mean": mean_val,
            "std": std_val,
            "rms": rms_val,
            "peak": peak_val,
            "crest_factor": crest_factor,
            "kurtosis": kurt_val,
            "skewness": skew_val,
            "p2p": p2p_val,
            "fft_mean": fft_mean,
            "fft_std": fft_std,
            "fft_peak": fft_peak,
            "dominant_freq": dominant_freq,
            "label": label
        }
        extracted_features.append(feature_row)
        
    features_df = pd.DataFrame(extracted_features)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"[*] Feature extraction complete. Saved to {features_path} (Shape: {features_df.shape})")
    return features_path

def scale_features(features_path="data/processed/features.csv", 
                   scaled_path="data/processed/features_scaled.csv", 
                   scaler_path="models/scaler.pkl"):
    """
    Separates the label, scales the features using StandardScaler,
    saves the fitted scaler, and saves the scaled features.
    """
    print(f"[*] Scaling features from {features_path}...")
    df = pd.read_csv(features_path)
    
    # Separate label
    y = df["label"]
    X = df.drop(columns=["label"])
    
    # Fit and transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[*] Fitted scaler saved to {scaler_path}")
    
    # Combine scaled features and label back
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df["label"] = y
    
    # Save the scaled features
    os.makedirs(os.path.dirname(scaled_path), exist_ok=True)
    scaled_df.to_csv(scaled_path, index=False)
    print(f"[*] Scaled features saved to {scaled_path} (Shape: {scaled_df.shape})")
    return scaled_path

if __name__ == "__main__":
    extract_features()
    scale_features()
