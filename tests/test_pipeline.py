import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

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
        
        # Build a 2-row DataFrame representing 2 Catenaria readings
        # Features: Altezza, Taglia, Temperatura, Umidita, Vento
        self.mock_df = pd.DataFrame([
            ["20/05/2026 T15:00:00", 1.2, 0.5, 22.4, 55.2, 12.5, 0],
            ["20/05/2026 T15:01:00", 1.3, 0.6, 23.1, 54.8, 14.1, 1]
        ], columns=["Data", "Altezza", "Taglia", "Temperatura", "Umidita", "Vento", "label"])
        
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
        
        # Verify shape (2 rows, 5 features + 1 label column = 6 columns)
        self.assertEqual(features_df.shape, (2, 6))
        
        # Get extracted features for normal window
        row = features_df.iloc[0]
        
        # Verify core Catenaria features match (and Data was dropped)
        self.assertNotIn("Data", features_df.columns)
        self.assertAlmostEqual(row["Altezza"], 1.2, places=5)
        self.assertAlmostEqual(row["Taglia"], 0.5, places=5)
        self.assertAlmostEqual(row["Temperatura"], 22.4, places=5)
        self.assertAlmostEqual(row["Umidita"], 55.2, places=5)
        self.assertAlmostEqual(row["Vento"], 12.5, places=5)

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
        self.assertEqual(len(scaler.mean_), 5)  # Should have 5 feature means

    def test_serialized_models(self):
        """
        Verifies that trained baseline model binaries can be successfully loaded and run inference.
        """
        # 1. Scaler Check
        scaler_file = "models/scaler.pkl"
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            self.assertEqual(scaler.n_features_in_, 5)
            
        # 2. Isolation Forest Model Check
        if_model_file = "models/isolation_forest/if_model.pkl"
        if os.path.exists(if_model_file):
            if_model = joblib.load(if_model_file)
            # Feed dummy sample (1 row, 5 features)
            dummy_sample = np.zeros((1, 5))
            prediction = if_model.predict(dummy_sample)
            self.assertIn(prediction[0], [1, -1])  # 1 = Normal, -1 = Anomaly
            
        # 3. XGBoost Model Check
        xgb_model_file = "models/xgboost/xgb_model.json"
        if os.path.exists(xgb_model_file):
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            xgb.load_model(xgb_model_file)
            # XGBoost expects 7 features (5 base scaled features + if_score + lstm_error)
            dummy_sample = np.zeros((1, 7))
            pred_class = xgb.predict(dummy_sample)
            self.assertIn(pred_class[0], [0, 1])  # 0 = Normal, 1 = Anomaly

if __name__ == "__main__":
    unittest.main()
