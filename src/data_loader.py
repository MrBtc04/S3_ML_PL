import os
import urllib.request
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm

# Mapping of local filenames to CWRU file numbers and labels
DATA_CONFIG = {
    "normal.mat": {"id": "97", "label": 0, "desc": "Normal Baseline"},
    "B007.mat": {"id": "118", "label": 1, "desc": "Ball Fault 0.007\""},
    "IR007.mat": {"id": "105", "label": 2, "desc": "Inner Race Fault 0.007\""},
    "OR007@6.mat": {"id": "130", "label": 3, "desc": "Outer Race Fault 0.007\" at 6:00"},
    "B021.mat": {"id": "222", "label": 1, "desc": "Ball Fault 0.021\""},
    "IR021.mat": {"id": "209", "label": 2, "desc": "Inner Race Fault 0.021\""},
    "OR021@6.mat": {"id": "234", "label": 3, "desc": "Outer Race Fault 0.021\" at 6:00"}
}

BASE_URL = "https://engineering.case.edu/sites/default/files"

def download_cwru_data(raw_dir="data/raw"):
    """
    Downloads the CWRU bearing dataset .mat files if they are not already present.
    """
    os.makedirs(raw_dir, exist_ok=True)
    print(f"[*] Ensuring CWRU dataset files are present in {raw_dir}...")
    
    for filename, info in DATA_CONFIG.items():
        dest_path = os.path.join(raw_dir, filename)
        if os.path.exists(dest_path):
            print(f"    - {filename} already exists. Skipping download.")
            continue
            
        file_id = info["id"]
        url = f"{BASE_URL}/{file_id}.mat"
        print(f"    - Downloading {filename} from {url}...")
        
        try:
            # Custom reporthook to show progress bar using tqdm
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                def reporthook(blocknum, blocksize, totalsize):
                    t.total = totalsize
                    t.update(blocknum * blocksize - t.n)
                
                # Bypassing potential SSL validation issues
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, context=ctx) as response, open(dest_path, 'wb') as out_file:
                    totalsize = int(response.info().get('Content-Length', 0))
                    t.total = totalsize
                    blocksize = 1024 * 8
                    read_bytes = 0
                    while True:
                        block = response.read(blocksize)
                        if not block:
                            break
                        out_file.write(block)
                        read_bytes += len(block)
                        t.update(len(block))
                        
            print(f"    - Successfully saved to {dest_path}")
        except Exception as e:
            print(f"[!] Error downloading {filename}: {e}")
            raise e

def load_cwru_data(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Loads raw CWRU bearing data files, extracts the drive-end (DE) signal,
    assigns labels, and saves the concatenated raw signals as a CSV.
    """
    download_cwru_data(raw_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    combined_signals = []
    print("[*] Processing raw CWRU files...")
    
    for filename, info in DATA_CONFIG.items():
        file_path = os.path.join(raw_dir, filename)
        label = info["label"]
        print(f"    - Reading {filename} (Label: {label}, Description: {info['desc']})...")
        
        try:
            mat_data = scipy.io.loadmat(file_path)
            # Find key containing "DE_time"
            de_key = [k for k in mat_data.keys() if "DE_time" in k]
            if not de_key:
                # If no DE_time key, try FE_time as backup or throw error
                raise KeyError(f"No Drive End (DE_time) signal found in keys: {list(mat_data.keys())}")
            
            de_key = de_key[0]
            # Flatten to 1D
            signal = mat_data[de_key].flatten()
            
            df = pd.DataFrame({
                "signal": signal,
                "label": label
            })
            combined_signals.append(df)
            print(f"      Loaded {len(signal)} samples from key '{de_key}'")
        except Exception as e:
            print(f"[!] Error loading {filename}: {e}")
            raise e
            
    print("[*] Concatenating and saving raw signals...")
    raw_df = pd.concat(combined_signals, ignore_index=True)
    out_path = os.path.join(processed_dir, "raw_signals.csv")
    raw_df.to_csv(out_path, index=False)
    print(f"[*] Raw signals saved to {out_path} (Shape: {raw_df.shape})")
    return out_path

def create_windows(raw_dir="data/raw", processed_dir="data/processed", window_size=1024, step_size=512):
    """
    Splits the vibration signals from each CWRU file into overlapping windows of size 1024
    with a step size of 512. Saves the matrix of windows to windows.csv.
    """
    os.makedirs(processed_dir, exist_ok=True)
    print(f"[*] Segmenting signals into windows (size: {window_size}, step: {step_size})...")
    
    all_windows = []
    
    for filename, info in DATA_CONFIG.items():
        file_path = os.path.join(raw_dir, filename)
        label = info["label"]
        print(f"    - Segmenting {filename}...")
        
        try:
            mat_data = scipy.io.loadmat(file_path)
            de_key = [k for k in mat_data.keys() if "DE_time" in k][0]
            signal = mat_data[de_key].flatten()
            
            num_samples = len(signal)
            file_windows = []
            
            for start_idx in range(0, num_samples - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window = signal[start_idx:end_idx]
                file_windows.append(window)
                
            print(f"      Created {len(file_windows)} windows from {num_samples} samples")
            
            # Create a DataFrame for this file's windows
            # Columns: w_0, w_1, ..., w_1023, label
            cols = [f"w_{i}" for i in range(window_size)]
            file_df = pd.DataFrame(file_windows, columns=cols)
            file_df["label"] = label
            
            all_windows.append(file_df)
        except Exception as e:
            print(f"[!] Error segmenting {filename}: {e}")
            raise e
            
    print("[*] Concatenating and saving windows...")
    windows_df = pd.concat(all_windows, ignore_index=True)
    out_path = os.path.join(processed_dir, "windows.csv")
    
    # Save as CSV
    windows_df.to_csv(out_path, index=False)
    print(f"[*] Windows saved to {out_path} (Shape: {windows_df.shape})")
    return out_path

if __name__ == "__main__":
    load_cwru_data()
    create_windows()
