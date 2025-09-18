# Firestore-Only Setup Guide

## 🎯 **What You Need for Firestore-Only Setup**

Since you're using **only Firestore** (not Firebase Authentication), you need:

### ✅ **Required:**
1. **Google Cloud Project** with Firestore enabled
2. **Firestore Service Account JSON** file
3. **JWT Secret Key** for your own authentication

### ❌ **NOT Required:**
- Firebase Authentication
- Firebase client SDK
- Connection strings
- Firebase project configuration

## 🔧 **Setup Steps**

### 1. **Enable Firestore in Google Cloud Console**
```bash
# Enable Firestore API
gcloud services enable firestore.googleapis.com
```

### 2. **Create Service Account**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Create a new service account with **Firestore User** role
4. Download the JSON key file
5. Place it in `config/firebase-credentials/`

### 3. **Set Environment Variables**
```bash
# Copy the example file
cp env.example .env

# Edit .env with your values
JWT_SECRET_KEY=your-super-secret-jwt-key-here
FIREBASE_SERVICE_ACCOUNT_PATH=config/firebase-credentials/your-service-account.json
```

### 4. **Test Locally**
```bash
python main.py
```

## 🚀 **Cloud Run Deployment**

### **Option 1: Using the Deploy Script**
```bash
./deploy.sh
```

### **Option 2: Manual Deployment**
```bash
# Set your project
gcloud config set project icc-rag-project

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com firestore.googleapis.com secretmanager.googleapis.com

# Create JWT secret
echo "your-jwt-secret-here" | gcloud secrets create jwt-secret-key --data-file=-

# Build and deploy
gcloud builds submit --tag gcr.io/icc-rag-project/icc-chatbot
gcloud run deploy icc-chatbot \
  --image gcr.io/icc-rag-project/icc-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="JWT_SECRET_KEY=jwt-secret-key:latest" \
  --set-env-vars="FIREBASE_SERVICE_ACCOUNT_PATH=/app/config/firebase-credentials/icc-project-472009-firebase-adminsdk.json"
```

## 🔐 **Secrets Management**

### **Required Secrets:**
1. **JWT_SECRET_KEY** - Strong secret for JWT token signing
2. **Firestore Service Account** - JSON file (included in container)

### **No Connection String Needed!**
Firestore uses the service account JSON file for authentication. No connection strings required.

## 📁 **File Structure for Firestore-Only**

```
config/
├── firebase-credentials/
│   └── icc-project-472009-firebase-adminsdk.json  # Service account
└── settings.py                                     # Configuration

backend/services/
├── firestore_auth.py                               # Firestore-only auth
└── auth_service.py                                 # JWT service
```

## ✅ **What Works with Firestore-Only**

- ✅ User registration and login
- ✅ Password hashing with bcrypt
- ✅ JWT token authentication
- ✅ Conversation storage
- ✅ User profile management
- ✅ All API endpoints

## 🚫 **What's NOT Used**

- ❌ Firebase Authentication SDK
- ❌ Firebase client-side libraries
- ❌ Firebase project configuration
- ❌ Connection strings
- ❌ Firebase Auth UI

## 🔍 **Troubleshooting**

### **"Firebase service account not found"**
- Check that your service account JSON is in the correct path
- Verify the `FIREBASE_SERVICE_ACCOUNT_PATH` environment variable

### **"Permission denied" errors**
- Ensure your service account has **Firestore User** role
- Check that Firestore API is enabled in your project

### **Authentication errors**
- Verify your JWT secret key is set correctly
- Check that the service account JSON is valid

## 🎉 **Summary**

Your current setup is already **Firestore-only**! You just need:
1. A valid service account JSON file
2. A JWT secret key
3. Firestore enabled in your Google Cloud project

No Firebase Authentication or connection strings needed!
