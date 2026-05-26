import os
import shutil
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_cwru_data, create_windows
from feature_engineering import extract_features, scale_features
from model_isolation_forest import train_isolation_forest
from model_lstm_autoencoder import train_lstm_autoencoder, create_sequences
from model_xgboost import train_xgboost
from ai_report import AIReport

class AutoTrainer:
    """
    Autoadestramento (Auto-training / Self-training): Retrains the full S3 ML Catenaria
    model suite, compares new validation performance with the currently deployed active models,
    archives previous models, and promotes new models if performance is superior or equal.
    """
    def __init__(self, workspace_dir="/Users/mariopaaris/Downloads/s3_ml_pipeline"):
        self.workspace_dir = workspace_dir
        self.models_dir = os.path.join(workspace_dir, "models")
        self.archive_dir = os.path.join(self.models_dir, "archive")
        self.reports_dir = os.path.join(workspace_dir, "output/reports")
        self.log_path = os.path.join(self.reports_dir, "autotrain_log.json")
        
    def run_autotrain(self, force_promote=False):
        """
        Runs the auto-training lifecycle.
        """
        print("\n" + "="*60)
        print("AUTO-TRAINING ENGINE STARTED")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_model_dir = os.path.join(self.models_dir, f"temp_retrain_{timestamp}")
        os.makedirs(temp_model_dir, exist_ok=True)
        
        try:
            # 1. Check if we have active models already. If not, we must force promote.
            active_exists = self._check_active_models_exist()
            if not active_exists:
                print("[!] No active models found in production. Promotion will be forced.")
                force_promote = True
                
            # 2. Evaluate current active models (baseline accuracy)
            old_accuracy = 0.0
            if active_exists:
                print("[*] Evaluating active production models on current validation set...")
                old_accuracy = self._evaluate_model_suite(
                    scaler_path=os.path.join(self.models_dir, "scaler.pkl"),
                    if_path=os.path.join(self.models_dir, "isolation_forest/if_model.pkl"),
                    lstm_path=os.path.join(self.models_dir, "lstm_autoencoder/lstm_ae_model.keras"),
                    xgb_path=os.path.join(self.models_dir, "xgboost/xgb_model.json")
                )
                print(f"    - Current Active Model Suite Accuracy: {old_accuracy:.6f}")
                
            # 3. Trigger retraining into a temporary directory
            print("[*] Retraining models on current dataset...")
            temp_scaler_path = os.path.join(temp_model_dir, "scaler.pkl")
            temp_if_path = os.path.join(temp_model_dir, "if_model.pkl")
            temp_lstm_path = os.path.join(temp_model_dir, "lstm_ae_model.keras")
            temp_xgb_path = os.path.join(temp_model_dir, "xgb_model.json")
            
            # Step A: Load, window, and extract features to ensure fresh inputs
            load_cwru_data()
            create_windows()
            extract_features()
            scale_features(scaler_path=temp_scaler_path)
            
            # Step B: Train Individual Models
            train_isolation_forest(model_path=temp_if_path)
            train_lstm_autoencoder(model_path=temp_lstm_path)
            train_xgboost(model_path=temp_xgb_path)
            
            # 4. Evaluate newly trained model suite
            print("[*] Evaluating newly retrained model suite...")
            new_accuracy = self._evaluate_model_suite(
                scaler_path=temp_scaler_path,
                if_path=temp_if_path,
                lstm_path=temp_lstm_path,
                xgb_path=temp_xgb_path
            )
            print(f"    - Retrained Model Suite Accuracy: {new_accuracy:.6f}")
            
            # 5. Promotion Logic
            promoted = False
            # Promote if performance is equal or better, or if forced
            if force_promote or (new_accuracy >= old_accuracy - 1e-5):
                print("[*] Promotion Criteria Met! Proceeding with model promotion...")
                promoted = True
                
                # A. Archive existing models
                if active_exists:
                    archive_path = os.path.join(self.archive_dir, timestamp)
                    os.makedirs(archive_path, exist_ok=True)
                    print(f"    - Archiving active models to {archive_path}...")
                    self._backup_active_models(archive_path)
                    
                # B. Promote new models
                print("    - Promoting retrained models to production paths...")
                self._promote_temp_models(temp_model_dir)
                
                # C. Run comparative evaluation to update active plots/csvs
                from evaluate import compare_models
                compare_models()
                
                # D. Regenerate AI Report
                reporter = AIReport()
                reporter.generate_report()
            else:
                print("[!] Retrained models did not outperform active production models. Promotion aborted.")
                
            # 6. Save Retraining Log
            log_entry = {
                "timestamp": timestamp,
                "active_models_evaluated": active_exists,
                "old_accuracy": round(old_accuracy, 6),
                "new_accuracy": round(new_accuracy, 6),
                "promoted": promoted,
                "forced": force_promote
            }
            self._save_log(log_entry)
            
            # Clean up temp folder
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
                
            print("="*60)
            print("AUTO-TRAINING PIPELINE COMPLETE")
            print("="*60 + "\n")
            return log_entry
            
        except Exception as e:
            print(f"[CRITICAL ERROR] Auto-training pipeline failed: {e}")
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
            raise e

    def _check_active_models_exist(self):
        required_paths = [
            os.path.join(self.models_dir, "scaler.pkl"),
            os.path.join(self.models_dir, "isolation_forest/if_model.pkl"),
            os.path.join(self.models_dir, "lstm_autoencoder/lstm_ae_model.keras"),
            os.path.join(self.models_dir, "xgboost/xgb_model.json")
        ]
        return all([os.path.exists(p) for p in required_paths])

    def _evaluate_model_suite(self, scaler_path, if_path, lstm_path, xgb_path):
        """
        Evaluates the model suite accuracy on the current validation split.
        """
        features_path = os.path.join(self.workspace_dir, "data/processed/features.csv")
        df = pd.read_csv(features_path)
        
        # Load scaler and transform base features
        scaler = joblib.load(scaler_path)
        base_features = ["Altezza", "Taglia", "Temperatura", "Umidita", "Vento"]
        X_scaled = scaler.transform(df[base_features])
        df_scaled = pd.DataFrame(X_scaled, columns=base_features)
        
        # Load and run Isolation Forest
        if_model = joblib.load(if_path)
        df_scaled["if_score"] = if_model.score_samples(X_scaled)
        df_scaled["if_prediction"] = if_model.predict(X_scaled)
        
        # Load and run LSTM Autoencoder
        # To avoid heavy imports unless needed
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(lstm_path)
        
        # Reshape for LSTM (3D: samples, timesteps, features) matching time_steps = 30
        time_steps = 30
        X_lstm = create_sequences(X_scaled, time_steps=time_steps)
        reconstructed = lstm_model.predict(X_lstm)
        # Reconstruction error (MAE)
        reconstruction_error_seq = np.mean(np.abs(X_lstm - reconstructed), axis=(1, 2))
        
        # Align sequence results back to original rows
        reconstruction_error = np.zeros(len(df))
        reconstruction_error[time_steps - 1:] = reconstruction_error_seq
        df_scaled["lstm_error"] = reconstruction_error
        
        # Load and run XGBoost Classifier
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        
        feature_cols = base_features + ["if_score", "lstm_error"]
        X_xgb = df_scaled[feature_cols]
        y = df["label"]
        
        # Stratified validation split consistent with standard training
        _, X_test, _, y_test = train_test_split(
            X_xgb, y, test_size=0.20, stratify=y, random_state=42
        )
        
        preds = xgb.predict(X_test)
        return accuracy_score(y_test, preds)

    def _backup_active_models(self, dest_path):
        os.makedirs(os.path.join(dest_path, "isolation_forest"), exist_ok=True)
        os.makedirs(os.path.join(dest_path, "lstm_autoencoder"), exist_ok=True)
        os.makedirs(os.path.join(dest_path, "xgboost"), exist_ok=True)
        
        shutil.copy(os.path.join(self.models_dir, "scaler.pkl"), os.path.join(dest_path, "scaler.pkl"))
        shutil.copy(os.path.join(self.models_dir, "isolation_forest/if_model.pkl"), os.path.join(dest_path, "isolation_forest/if_model.pkl"))
        shutil.copy(os.path.join(self.models_dir, "lstm_autoencoder/lstm_ae_model.keras"), os.path.join(dest_path, "lstm_autoencoder/lstm_ae_model.keras"))
        shutil.copy(os.path.join(self.models_dir, "xgboost/xgb_model.json"), os.path.join(dest_path, "xgboost/xgb_model.json"))

    def _promote_temp_models(self, temp_path):
        # Create necessary folders if not exists
        os.makedirs(os.path.join(self.models_dir, "isolation_forest"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "lstm_autoencoder"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "xgboost"), exist_ok=True)
        
        shutil.copy(os.path.join(temp_path, "scaler.pkl"), os.path.join(self.models_dir, "scaler.pkl"))
        shutil.copy(os.path.join(temp_path, "if_model.pkl"), os.path.join(self.models_dir, "isolation_forest/if_model.pkl"))
        shutil.copy(os.path.join(temp_path, "lstm_ae_model.keras"), os.path.join(self.models_dir, "lstm_autoencoder/lstm_ae_model.keras"))
        shutil.copy(os.path.join(temp_path, "xgb_model.json"), os.path.join(self.models_dir, "xgboost/xgb_model.json"))

    def _save_log(self, entry):
        os.makedirs(self.reports_dir, exist_ok=True)
        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
                
        logs.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=4)

if __name__ == "__main__":
    trainer = AutoTrainer()
    trainer.run_autotrain()
