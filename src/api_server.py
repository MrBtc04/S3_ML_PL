import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from auto_training import AutoTrainer
from ai_report import AIReport

# Initialize the FastAPI app
app = FastAPI(
    title="S3 ML Catenaria Pipeline API",
    description="REST API for accessing anomaly detection reports, telemetry analysis, and automated model retraining.",
    version="1.0.0"
)

# Enable CORS for standard web client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_DIR = "/Users/mariopaaris/Downloads/s3_ml_pipeline"
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "output/reports")
PLOTS_DIR = os.path.join(WORKSPACE_DIR, "output/plots")

# Pipeline task status
pipeline_status = {
    "is_running": False,
    "last_run_timestamp": None,
    "last_run_status": None,
    "error_message": None
}

def run_pipeline_task():
    global pipeline_status
    pipeline_status["is_running"] = True
    pipeline_status["error_message"] = None
    try:
        print("[*] Background Pipeline Task started...")
        from data_loader import load_cwru_data, create_windows
        from feature_engineering import extract_features, scale_features
        from model_isolation_forest import train_isolation_forest
        from model_lstm_autoencoder import train_lstm_autoencoder
        from model_xgboost import train_xgboost
        from evaluate import compare_models
        
        load_cwru_data()
        create_windows()
        extract_features()
        scale_features()
        train_isolation_forest()
        train_lstm_autoencoder()
        train_xgboost()
        compare_models()
        
        # Regenerate report
        reporter = AIReport()
        reporter.generate_report()
        
        pipeline_status["last_run_status"] = "SUCCESS"
        print("[*] Background Pipeline Task completed successfully.")
    except Exception as e:
        pipeline_status["last_run_status"] = "FAILED"
        pipeline_status["error_message"] = str(e)
        print(f"[!] Background Pipeline Task failed: {e}")
    finally:
        import time
        pipeline_status["is_running"] = False
        pipeline_status["last_run_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

def run_autotrain_task():
    global pipeline_status
    pipeline_status["is_running"] = True
    pipeline_status["error_message"] = None
    try:
        print("[*] Background Auto-training Task started...")
        trainer = AutoTrainer(WORKSPACE_DIR)
        trainer.run_autotrain()
        pipeline_status["last_run_status"] = "SUCCESS"
        print("[*] Background Auto-training Task completed successfully.")
    except Exception as e:
        pipeline_status["last_run_status"] = "FAILED"
        pipeline_status["error_message"] = str(e)
        print(f"[!] Background Auto-training Task failed: {e}")
    finally:
        import time
        pipeline_status["is_running"] = False
        pipeline_status["last_run_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.get("/health", summary="Get service and model status")
def get_health():
    """
    Returns the status of the service and checks if the active model binaries exist.
    """
    models = {
        "scaler": os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")),
        "isolation_forest": os.path.exists(os.path.join(MODELS_DIR, "isolation_forest/if_model.pkl")),
        "lstm_autoencoder": os.path.exists(os.path.join(MODELS_DIR, "lstm_autoencoder/lstm_ae_model.keras")),
        "xgboost": os.path.exists(os.path.join(MODELS_DIR, "xgboost/xgb_model.json"))
    }
    
    status = "OK" if all(models.values()) else "INCOMPLETE_MODELS"
    
    return {
        "status": status,
        "pipeline_state": pipeline_status,
        "models_presence": models
    }

@app.get("/api/report/latest", summary="Retrieve latest AI diagnostic report data")
def get_latest_report():
    """
    Returns the JSON representation of the latest AI diagnostic report.
    """
    report_json_path = os.path.join(REPORTS_DIR, "ai_report.json")
    if not os.path.exists(report_json_path):
        # Try generating it if files are ready
        scaled_features = os.path.join(WORKSPACE_DIR, "data/processed/features_scaled.csv")
        if os.path.exists(scaled_features):
            try:
                reporter = AIReport()
                reporter.generate_report()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate report on the fly: {e}")
        else:
            raise HTTPException(
                status_code=404, 
                detail="No diagnostic report found. Please run the pipeline first using '/api/pipeline/run'."
            )
            
    try:
        with open(report_json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {e}")

@app.get("/api/report/download", summary="Download Markdown or HTML report")
def download_report(file_format: str = "html"):
    """
    Downloads or views the diagnostic report.
    Allowed formats: 'html', 'md'.
    """
    fmt = file_format.lower().strip()
    if fmt not in ["html", "md"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats are 'html' and 'md'.")
        
    filename = f"ai_report.{fmt}"
    file_path = os.path.join(REPORTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Report file {filename} not found. Run pipeline first.")
        
    media_types = {
        "html": "text/html",
        "md": "text/markdown"
    }
    
    return FileResponse(
        path=file_path,
        media_type=media_types[fmt],
        filename=filename
    )

@app.get("/api/plots/{plot_name}", summary="Serve generated diagnostic plots")
def get_plot(plot_name: str):
    """
    Serves a PNG plot from the output plots directory.
    Example plot names: 'model_comparison.png', 'confusion_matrix.png', 'feature_importance.png'.
    """
    clean_name = os.path.basename(plot_name)
    plot_path = os.path.join(PLOTS_DIR, clean_name)
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail=f"Plot '{clean_name}' not found.")
        
    return FileResponse(path=plot_path, media_type="image/png")

@app.post("/api/pipeline/run", summary="Trigger a full pipeline execution")
def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers a full execution of the CWRU anomaly detection pipeline.
    Runs asynchronously in the background.
    """
    if pipeline_status["is_running"]:
        return JSONResponse(
            status_code=409, 
            content={"message": "Pipeline execution or auto-training is already in progress."}
        )
        
    background_tasks.add_task(run_pipeline_task)
    return {"message": "Pipeline run triggered successfully in the background. Monitor status via '/health'."}

@app.post("/api/pipeline/autotrain", summary="Trigger the self-training workflow")
def trigger_autotrain(background_tasks: BackgroundTasks):
    """
    Triggers the self-training / automated retraining pipeline.
    Runs asynchronously in the background.
    """
    if pipeline_status["is_running"]:
        return JSONResponse(
            status_code=409, 
            content={"message": "Pipeline execution or auto-training is already in progress."}
        )
        
    background_tasks.add_task(run_autotrain_task)
    return {"message": "Auto-training retraining workflow triggered successfully. Monitor status via '/health'."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
