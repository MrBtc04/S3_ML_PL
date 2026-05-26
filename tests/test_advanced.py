import os
import sys
import json
import shutil
import unittest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure src/ is in the python search path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_report import AIReport
from auto_training import AutoTrainer
from api_server import app

class TestAdvancedFeatures(unittest.TestCase):
    
    def setUp(self):
        """
        Setup sandboxed paths and back up real files.
        """
        self.workspace_dir = "/Users/mariopaaris/Downloads/s3_ml_pipeline"
        self.scaled_path = os.path.join(self.workspace_dir, "data/processed/features_scaled.csv")
        self.reports_dir = os.path.join(self.workspace_dir, "output/reports")
        self.models_dir = os.path.join(self.workspace_dir, "models")
        
        # Backup paths
        self.scaled_backup = self.scaled_path + ".backup"
        self.report_backups = {}
        
        # 1. Back up existing features_scaled.csv
        if os.path.exists(self.scaled_path):
            shutil.copy(self.scaled_path, self.scaled_backup)
            
        # 2. Back up existing reports
        for ext in ["json", "md", "html"]:
            rep_path = os.path.join(self.reports_dir, f"ai_report.{ext}")
            if os.path.exists(rep_path):
                self.report_backups[ext] = rep_path + ".backup"
                shutil.copy(rep_path, self.report_backups[ext])
                
        # 3. Create sandboxed mock features_scaled.csv with all predicted columns
        os.makedirs(os.path.dirname(self.scaled_path), exist_ok=True)
        mock_data = {
            "label": [0, 1, 0, 1],
            "if_prediction": [1, 1, -1, -1],
            "if_score": [0.1, 0.05, -0.15, -0.25],
            "lstm_anomaly": ["No", "No", "Yes", "Yes"],
            "lstm_error": [0.05, 0.08, 0.35, 0.45],
            "xgb_prediction": [0, 1, 0, 1],
            "Altezza": [1.2, 1.3, 1.2, 1.3],
            "Taglia": [0.5, 0.6, 0.5, 0.6],
            "Temperatura": [22.4, 23.1, 22.4, 23.1],
            "Umidita": [55.2, 54.8, 55.2, 54.8],
            "Vento": [12.5, 14.1, 12.5, 14.1]
        }
        pd.DataFrame(mock_data).to_csv(self.scaled_path, index=False)
        
        # FastAPI client for API routing tests
        self.client = TestClient(app)

    def tearDown(self):
        """
        Restore real files from backups and clean up.
        """
        # 1. Restore features_scaled.csv
        if os.path.exists(self.scaled_backup):
            shutil.move(self.scaled_backup, self.scaled_path)
        elif os.path.exists(self.scaled_path):
            os.remove(self.scaled_path)
            
        # 2. Restore reports
        for ext, backup_path in self.report_backups.items():
            rep_path = os.path.join(self.reports_dir, f"ai_report.{ext}")
            if os.path.exists(backup_path):
                shutil.move(backup_path, rep_path)
            elif os.path.exists(rep_path):
                os.remove(rep_path)

    def test_ai_report_generation(self):
        """
        Checks if AIReport successfully creates JSON, MD, and HTML files.
        """
        reporter = AIReport(scaled_path=self.scaled_path, reports_dir=self.reports_dir)
        reporter.generate_report()
        
        # Verify files are successfully created
        json_path = os.path.join(self.reports_dir, "ai_report.json")
        md_path = os.path.join(self.reports_dir, "ai_report.md")
        html_path = os.path.join(self.reports_dir, "ai_report.html")
        
        self.assertTrue(os.path.exists(json_path), "JSON report should be generated")
        self.assertTrue(os.path.exists(md_path), "Markdown report should be generated")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")
        
        # Verify JSON report structure
        with open(json_path, "r") as f:
            data = json.load(f)
            self.assertIn("metrics", data)
            self.assertIn("insights", data)
            self.assertIn("recommendations", data)
            
            metrics = data["metrics"]
            self.assertIn("system_health_state", metrics)
            self.assertIn("severity_level", metrics)
            self.assertIn("high_confidence_anomalies", metrics)
            
    def test_auto_trainer_utilities(self):
        """
        Verifies model presence checks and archiving capabilities.
        """
        trainer = AutoTrainer(self.workspace_dir)
        
        # 1. Check model presence detection matches local state
        active_exists = trainer._check_active_models_exist()
        
        required_paths = [
            os.path.join(self.models_dir, "scaler.pkl"),
            os.path.join(self.models_dir, "isolation_forest/if_model.pkl"),
            os.path.join(self.models_dir, "lstm_autoencoder/lstm_ae_model.keras"),
            os.path.join(self.models_dir, "xgboost/xgb_model.json")
        ]
        expected_existence = all([os.path.exists(p) for p in required_paths])
        self.assertEqual(active_exists, expected_existence)
        
        # 2. Test backup mechanism
        if active_exists:
            test_backup_dir = os.path.join(self.workspace_dir, "models/archive/test_backup")
            if os.path.exists(test_backup_dir):
                shutil.rmtree(test_backup_dir)
                
            trainer._backup_active_models(test_backup_dir)
            
            # Check backup copies
            self.assertTrue(os.path.exists(os.path.join(test_backup_dir, "scaler.pkl")))
            self.assertTrue(os.path.exists(os.path.join(test_backup_dir, "isolation_forest/if_model.pkl")))
            self.assertTrue(os.path.exists(os.path.join(test_backup_dir, "lstm_autoencoder/lstm_ae_model.keras")))
            self.assertTrue(os.path.exists(os.path.join(test_backup_dir, "xgboost/xgb_model.json")))
            
            # Clean up backup
            shutil.rmtree(test_backup_dir)

    def test_fastapi_endpoints(self):
        """
        Executes HTTP requests against FastAPI endpoints to test routing.
        """
        # Ensure a sandboxed report exists so we can query it
        reporter = AIReport(scaled_path=self.scaled_path, reports_dir=self.reports_dir)
        reporter.generate_report()
        
        # Test Health Check
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertIn("status", res_json)
        self.assertIn("models_presence", res_json)
        
        # Test Get Latest Report
        response_report = self.client.get("/api/report/latest")
        self.assertEqual(response_report.status_code, 200)
        report_data = response_report.json()
        self.assertIn("metrics", report_data)
            
        # Test Download HTML Report
        response_html = self.client.get("/api/report/download?file_format=html")
        self.assertEqual(response_html.status_code, 200)
        self.assertEqual(response_html.headers["content-type"], "text/html; charset=utf-8")
            
        # Test Download MD Report
        response_md = self.client.get("/api/report/download?file_format=md")
        self.assertEqual(response_md.status_code, 200)
        self.assertEqual(response_md.headers["content-type"], "text/markdown; charset=utf-8")

        # Test Invalid Format Download
        response_invalid = self.client.get("/api/report/download?file_format=invalid")
        self.assertEqual(response_invalid.status_code, 400)
        
        # Test trigger pipeline endpoint response
        response_run = self.client.post("/api/pipeline/run")
        self.assertEqual(response_run.status_code, 200)
        self.assertIn("triggered successfully", response_run.json()["message"])

if __name__ == "__main__":
    unittest.main()
