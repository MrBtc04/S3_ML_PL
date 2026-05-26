import os
import sys

# Add src directory to PYTHONPATH so imports resolve correctly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("STARTING S3 ML CATENARIA PIPELINE API SERVER")
    print("API Documentation available at: http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    
    # Run the Uvicorn ASGI server
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
