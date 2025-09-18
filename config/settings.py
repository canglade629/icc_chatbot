"""
Configuration settings for the ICC Legal Research Assistant
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database settings
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", 
    str(BASE_DIR / "config" / "firebase-credentials" / "icc-project-472009-firebase-adminsdk.json")
)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# CORS settings
CORS_ORIGINS = ["*"]  # Change in production
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Application settings
APP_NAME = "ICC Legal Research Assistant"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Data processing settings
MAX_CONVERSATIONS_PER_USER = 20
CONVERSATION_CLEANUP_BATCH_SIZE = 10

# File paths
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
COMPONENTS_DIR = FRONTEND_DIR / "components"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_processing" / "output"
