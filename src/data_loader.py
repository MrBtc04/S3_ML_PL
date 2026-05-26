import os
import shutil
import pandas as pd
import numpy as np

def load_cwru_data(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Loads raw Catenaria dataset CSV, cleans and pivots it to a wide format,
    creates the target 'label' column, and saves to data/processed/raw_signals.csv.
    """
    raw_path = os.path.join(raw_dir, "CatenariadatasetTraning.csv")
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"[*] Loading Catenaria raw dataset from {raw_path}...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Please place CatenariadatasetTraning.csv there.")
        
    # Read the raw CSV (semicolon separated)
    df = pd.read_csv(raw_path, sep=";")
    print(f"    - Raw data loaded successfully. Shape: {df.shape}")
    
    # Extract clean sensor name from Topic path
    df["sensor"] = df["Topic"].apply(lambda x: x.split("/")[-1])
    
    # Sort and deduplicate to resolve multiple values per (Data, sensor) pair
    # Sorting TEMPERATURA ascending ('INVALID' < 'VALID') so that dropping duplicates keeping 'last'
    # preserves the 'VALID' measurement if both exist.
    print("[*] Deduplicating and cleaning sensor data...")
    df_sorted = df.sort_values(by=["Data", "sensor", "TEMPERATURA"], ascending=[True, True, True])
    df_dedup = df_sorted.drop_duplicates(subset=["Data", "sensor"], keep="last")
    
    # Pivot to wide format
    print("[*] Reshaping data into wide multi-variable time series...")
    df_wide = df_dedup.pivot(index="Data", columns="sensor", values="Valore")
    
    # Calculate target anomaly label per timestamp
    # 1 (Anomaly) if ANY sensor record at that timestamp is 'INVALID', else 0
    print("[*] Constructing binary target anomaly labels...")
    df_label = df.groupby("Data")["TEMPERATURA"].apply(
        lambda x: 1 if "INVALID" in x.values else 0
    ).to_frame(name="label")
    
    # Merge pivoted features and target labels
    dataset = df_wide.join(df_label).reset_index()
    
    # Parse Data column as datetime for chronological sorting
    print("[*] Sorting dataset chronologically...")
    dataset["Datetime"] = pd.to_datetime(dataset["Data"], format="%d/%m/%Y T%H:%M:%S")
    dataset = dataset.sort_values(by="Datetime").drop(columns=["Datetime"])
    
    # Save pivoted and aligned dataset
    out_path = os.path.join(processed_dir, "raw_signals.csv")
    dataset.to_csv(out_path, index=False)
    print(f"[*] Aligned Catenaria dataset saved to {out_path} (Shape: {dataset.shape})")
    
    return out_path

def create_windows(raw_dir="data/raw", processed_dir="data/processed", window_size=1024, step_size=512):
    """
    Since the Catenaria dataset consists of minutely multi-variable sensor readings,
    high-frequency segmentation is not required. This function acts as a robust pass-through
    that copies raw_signals.csv to windows.csv to maintain pipeline structure compatibility.
    """
    os.makedirs(processed_dir, exist_ok=True)
    src_path = os.path.join(processed_dir, "raw_signals.csv")
    dest_path = os.path.join(processed_dir, "windows.csv")
    
    print(f"[*] Copying {src_path} to {dest_path} (pass-through)...")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Base file {src_path} not found. Ensure load_cwru_data has executed.")
        
    shutil.copy(src_path, dest_path)
    print(f"[*] Pass-through complete. Windows file saved to {dest_path}")
    
    return dest_path

if __name__ == "__main__":
    load_cwru_data()
    create_windows()
