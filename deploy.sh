#!/bin/bash

# ICC Legal Research Assistant - Cloud Run Deployment Script
# Firestore-only setup

set -e

# Configuration
PROJECT_ID="icc-rag-project"
SERVICE_NAME="icc-chatbot"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying ICC Legal Research Assistant to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create JWT secret if it doesn't exist
echo "🔐 Setting up JWT secret..."
if ! gcloud secrets describe jwt-secret-key >/dev/null 2>&1; then
    echo "Creating JWT secret..."
    echo "your-super-secret-jwt-key-change-in-production-$(date +%s)" | gcloud secrets create jwt-secret-key --data-file=-
else
    echo "JWT secret already exists"
fi

# Build and push the container
echo "🏗️ Building and pushing container..."
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-secrets="JWT_SECRET_KEY=jwt-secret-key:latest" \
    --set-env-vars="FIREBASE_SERVICE_ACCOUNT_PATH=/app/config/firebase-credentials/icc-project-472009-firebase-adminsdk.json" \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --port 8000

echo "✅ Deployment complete!"
echo "🌐 Your app is available at:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
