import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(windows_path="data/processed/windows.csv", features_path="data/processed/features.csv"):
    """
    Loads windows.csv, extracts raw sensor features (dropping non-numeric columns like Data),
    and saves to data/processed/features.csv.
    """
    print(f"[*] Extracting features from {windows_path}...")
    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"Windows file not found at {windows_path}. Has data_loader executed?")
        
    df = pd.read_csv(windows_path)
    
    # Drop string-based Data column if present to ensure purely numeric dataset
    if "Data" in df.columns:
        df = df.drop(columns=["Data"])
        
    # Columns remaining: Altezza, Taglia, Temperatura, Umidita, Vento, label
    print(f"    - Cleaned feature shape (excluding timestamp): {df.shape}")
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_csv(features_path, index=False)
    print(f"[*] Features extracted and saved to {features_path}")
    
    return features_path

def scale_features(features_path="data/processed/features.csv", 
                   scaled_path="data/processed/features_scaled.csv", 
                   scaler_path="models/scaler.pkl"):
    """
    Scales the numeric sensor features using StandardScaler, preserves the label column,
    and serializes the fitted scaler to models/scaler.pkl.
    """
    print(f"[*] Scaling features from {features_path}...")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}.")
        
    df = pd.read_csv(features_path)
    
    # Separate features and target label
    y = df["label"]
    X = df.drop(columns=["label"])
    
    # Fit and transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the fitted scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[*] Fitted scaler successfully saved to {scaler_path}")
    
    # Combine scaled features and label
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df["label"] = y
    
    # Save the scaled features DataFrame
    os.makedirs(os.path.dirname(scaled_path), exist_ok=True)
    scaled_df.to_csv(scaled_path, index=False)
    print(f"[*] Scaled features saved to {scaled_path} (Shape: {scaled_df.shape})")
    
    return scaled_path

if __name__ == "__main__":
    extract_features()
    scale_features()
