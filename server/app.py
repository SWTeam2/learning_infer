# app.py

from fastapi import FastAPI
import uvicorn
from app.app_routing import app as routing_app  # Import the FastAPI app instance from app_routing.py

if __name__ == "__main__":
    uvicorn.run(routing_app, host="0.0.0.0", port=48000)
