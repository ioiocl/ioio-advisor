import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def start_app():
    """Start the FastAPI application."""
    try:
        # Set OpenTelemetry configuration
        os.environ["OTEL_PYTHON_FASTAPI_EXCLUDED_URLS"] = "health"
        os.environ["OTEL_METRICS_EXPORTER"] = "prometheus"
        
        # Start the application
        uvicorn.run(
            "src.infrastructure.api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", 8000)),
            reload=os.getenv("DEBUG", "False").lower() == "true",
            workers=int(os.getenv("API_WORKERS", 1))
        )
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_app()
