import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib
from scipy.stats import kurtosis, skew

# Append src/ to path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import extract_features, scale_features

class TestMLPipeline(unittest.TestCase):
    
    def setUp(self):
        """
        Creates synthetic data and directories for localized module tests.
        """
        self.tmp_dir = "data/processed/tmp_test"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.mock_windows_path = os.path.join(self.tmp_dir, "mock_windows.csv")
        self.mock_features_path = os.path.join(self.tmp_dir, "mock_features.csv")
        self.mock_scaled_path = os.path.join(self.tmp_dir, "mock_features_scaled.csv")
        self.mock_scaler_path = os.path.join(self.tmp_dir, "mock_scaler.pkl")
        
        # Create a mock window of 1024 samples (sine wave + noise)
        np.random.seed(42)
        t = np.linspace(0, 1, 1024, endpoint=False)
        # 10 Hz sine wave + noise
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1024)
        
        # Build a 2-row DataFrame representing 2 windows
        cols = [f"w_{i}" for i in range(1024)]
        self.mock_df = pd.DataFrame([signal, -signal], columns=cols)
        self.mock_df["label"] = [0, 1]  # Row 0: Normal, Row 1: Fault
        
        self.mock_df.to_csv(self.mock_windows_path, index=False)

    def tearDown(self):
        """
        Cleans up temporary testing files.
        """
        for f in [self.mock_windows_path, self.mock_features_path, 
                  self.mock_scaled_path, self.mock_scaler_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_feature_calculations(self):
        """
        Verifies that feature engineering extracts exact, mathematically sound metrics.
        """
        # Run extraction on mock windows
        extract_features(windows_path=self.mock_windows_path, features_path=self.mock_features_path)
        
        self.assertTrue(os.path.exists(self.mock_features_path))
        features_df = pd.read_csv(self.mock_features_path)
        
        # Verify shape (2 windows, 12 features + 1 label column = 13 columns)
        self.assertEqual(features_df.shape, (2, 13))
        
        # Get extracted features for normal window
        row = features_df.iloc[0]
        window_raw = self.mock_df.drop(columns=["label"]).iloc[0].values
        
        # Verify core time-domain calculations
        self.assertAlmostEqual(row["mean"], np.mean(window_raw), places=5)
        self.assertAlmostEqual(row["std"], np.std(window_raw), places=5)
        self.assertAlmostEqual(row["rms"], np.sqrt(np.mean(window_raw**2)), places=5)
        self.assertAlmostEqual(row["peak"], np.max(np.abs(window_raw)), places=5)
        self.assertAlmostEqual(row["p2p"], np.max(window_raw) - np.min(window_raw), places=5)
        self.assertAlmostEqual(row["kurtosis"], kurtosis(window_raw), places=5)
        self.assertAlmostEqual(row["skewness"], skew(window_raw), places=5)

    def test_scaler_persistence(self):
        """
        Verifies that standard scaling runs and saves the fitted scaler object properly.
        """
        extract_features(windows_path=self.mock_windows_path, features_path=self.mock_features_path)
        
        # Run scaling
        scale_features(features_path=self.mock_features_path, 
                       scaled_path=self.mock_scaled_path, 
                       scaler_path=self.mock_scaler_path)
        
        self.assertTrue(os.path.exists(self.mock_scaled_path))
        self.assertTrue(os.path.exists(self.mock_scaler_path))
        
        # Load scaler and verify it's a valid object
        scaler = joblib.load(self.mock_scaler_path)
        self.assertTrue(hasattr(scaler, "mean_"))
        self.assertEqual(len(scaler.mean_), 12)  # Should have 12 feature means

    def test_serialized_models(self):
        """
        Verifies that trained baseline model binaries can be successfully loaded and run inference.
        """
        # 1. Scaler Check
        scaler_file = "models/scaler.pkl"
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.assertEqual(scaler.n_features_in_, 12)
            
        # 2. Isolation Forest Model Check
        if_model_file = "models/isolation_forest/if_model.pkl"
        if os.path.exists(if_model_file):
            if_model = joblib.load(if_model_file)
            # Feed dummy sample (1 row, 12 features)
            dummy_sample = np.zeros((1, 12))
            prediction = if_model.predict(dummy_sample)
            self.assertIn(prediction[0], [1, -1])  # 1 = Normal, -1 = Anomaly
            
        # 3. XGBoost Model Check
        xgb_model_file = "models/xgboost/xgb_model.json"
        if os.path.exists(xgb_model_file):
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            xgb.load_model(xgb_model_file)
            # XGBoost expects 14 features (12 base scaled features + if_score + lstm_error)
            dummy_sample = np.zeros((1, 14))
            pred_class = xgb.predict(dummy_sample)
            self.assertIn(pred_class[0], [0, 1, 2, 3])

if __name__ == "__main__":
    unittest.main()
