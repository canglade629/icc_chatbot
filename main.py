#!/usr/bin/env python3
"""
ICC Legal Research Assistant - Main Entry Point

This is the main entry point for the ICC Legal Research Assistant application.
It imports and runs the FastAPI application from the backend/api module.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from api.app import app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting ICC Legal Research Assistant on {host}:{port}")
    uvicorn.run(app, host=host, port=port)