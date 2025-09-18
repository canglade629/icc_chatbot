# Cloud Run Deployment - Firestore Only

## 🎯 **Quick Answer to Your Question**

**Do you need Firebase stuff?** ❌ **NO** - You only need Firestore
**Do you need a connection string?** ❌ **NO** - Firestore uses the service account JSON file

## 🚀 **What You Need for Cloud Run**

### **Required:**
1. ✅ **Google Cloud Project** (you have: `icc-rag-project`)
2. ✅ **Firestore Service Account JSON** (you have this)
3. ✅ **JWT Secret Key** (we'll create this)

### **NOT Required:**
- ❌ Firebase Authentication
- ❌ Connection strings
- ❌ Firebase project configuration

## 🔧 **Step-by-Step Deployment**

### **1. Enable Required APIs**
```bash
gcloud config set project icc-rag-project
gcloud services enable run.googleapis.com cloudbuild.googleapis.com firestore.googleapis.com secretmanager.googleapis.com
```

### **2. Create JWT Secret**
```bash
# Create a strong JWT secret
echo "your-super-secret-jwt-key-$(date +%s)" | gcloud secrets create jwt-secret-key --data-file=-
```

### **3. Deploy to Cloud Run**
```bash
# Build and push
gcloud builds submit --tag gcr.io/icc-rag-project/icc-chatbot

# Deploy
gcloud run deploy icc-chatbot \
  --image gcr.io/icc-rag-project/icc-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="JWT_SECRET_KEY=jwt-secret-key:latest" \
  --set-env-vars="FIREBASE_SERVICE_ACCOUNT_PATH=/app/config/firebase-credentials/icc-project-472009-firebase-adminsdk.json" \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10
```

## 🔐 **Secrets Configuration**

### **JWT Secret (Required)**
- **What it is:** Secret key for signing JWT tokens
- **Where:** Google Cloud Secret Manager
- **How:** Created via `gcloud secrets create`

### **Firestore Service Account (Required)**
- **What it is:** JSON file for Firestore authentication
- **Where:** Included in Docker container
- **How:** Already in your `config/firebase-credentials/` folder

## 📁 **Container Structure**

```
/app/
├── config/
│   └── firebase-credentials/
│       └── icc-project-472009-firebase-adminsdk.json  # Firestore auth
├── backend/
│   └── services/
│       └── firestore_auth.py                          # Firestore-only auth
└── frontend/
    └── index.html                                     # React app
```

## ✅ **What Works with This Setup**

- ✅ User registration and login
- ✅ Password hashing with bcrypt
- ✅ JWT token authentication
- ✅ Conversation storage in Firestore
- ✅ User profile management
- ✅ All API endpoints
- ✅ Responsive React frontend

## 🚫 **What's NOT Used**

- ❌ Firebase Authentication SDK
- ❌ Firebase client-side libraries
- ❌ Firebase project configuration
- ❌ Connection strings
- ❌ Firebase Auth UI

## 🔍 **Environment Variables in Cloud Run**

```bash
# Set via Cloud Run deployment
JWT_SECRET_KEY=jwt-secret-key:latest                    # From Secret Manager
FIREBASE_SERVICE_ACCOUNT_PATH=/app/config/firebase-credentials/icc-project-472009-firebase-adminsdk.json
```

## 🎉 **Summary**

Your setup is **already Firestore-only**! You just need:

1. **Enable APIs** in your Google Cloud project
2. **Create JWT secret** in Secret Manager
3. **Deploy to Cloud Run** with the service account JSON

**No Firebase Authentication needed!** 🎯
