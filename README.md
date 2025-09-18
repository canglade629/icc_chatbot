# ICC Legal Research Assistant

A comprehensive legal research assistant for International Criminal Court (ICC) documentation, built with FastAPI backend and React frontend, featuring user authentication, conversation management, and document processing capabilities.

## 🏗️ Project Structure

```
icc_chatbot/
├── backend/                    # Backend API and services
│   ├── api/                   # FastAPI application
│   │   └── app.py            # Main API application
│   ├── services/             # Business logic services
│   │   ├── auth_service.py   # JWT authentication service
│   │   ├── firestore_auth.py # Firestore authentication
│   │   ├── firebase_config.py # Firebase configuration
│   │   └── firebase_config_mock.py # Mock auth for testing
│   ├── models/               # Data models (future)
│   └── test_auth.py          # Authentication tests
├── frontend/                  # Frontend application
│   ├── static/               # Static assets
│   │   └── icc_logo.svg      # ICC logo
│   ├── components/           # React components
│   │   └── auth.html         # Authentication page
│   └── index.html            # Main React application
├── data_processing/          # Data processing and ML
│   ├── chunking/            # Document chunking scripts
│   │   ├── geneva_convention_chunker.py
│   │   ├── pdf_chunker.py
│   │   ├── section_classifier.py
│   │   └── test_*.py        # Chunking tests
│   ├── notebooks/           # Jupyter notebooks
│   │   └── ICC_Enhanced_RAG_Production.ipynb
│   ├── output/              # Processed data outputs
│   │   └── *.parquet        # Chunked data files
│   └── save/                # Additional data storage
├── config/                   # Configuration files
│   ├── settings.py          # Application settings
│   └── firebase-credentials/ # Firebase service account
├── scripts/                  # Utility scripts
│   └── explore_databricks_data.py
├── data/                     # Raw data files
│   └── AI IHL/              # International Humanitarian Law documents
│       ├── documentation/   # Geneva Conventions, Protocols
│       └── past_judgements/ # ICC case files
├── docs/                     # Documentation
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
└── README.md                # This file
```

## 🚀 Features

### Core Functionality
- **User Authentication**: Email/password with JWT tokens
- **Conversation Management**: Persistent chat history in Firestore
- **Document Processing**: PDF chunking and text extraction
- **Legal Research**: AI-powered responses for ICC documentation
- **Responsive Design**: Mobile and desktop optimized

### Authentication & Security
- JWT-based authentication with refresh tokens
- Password hashing with bcrypt
- Firestore integration for user data
- Session management and token validation

### Conversation Features
- Auto-generated conversation titles
- Timestamp display with smart formatting
- 20 conversation limit per user
- Empty conversation cleanup
- Scrollable conversation history

### Data Processing
- PDF document chunking
- Text extraction and preprocessing
- Section classification
- Parquet data storage
- Jupyter notebook analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js (for frontend development)
- Google Cloud Project with Firestore enabled
- Firebase service account credentials

### 1. Clone the Repository
```bash
git clone <repository-url>
cd icc_chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```env
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-here

# Firebase Configuration
FIREBASE_SERVICE_ACCOUNT_PATH=config/firebase-credentials/icc-project-472009-firebase-adminsdk.json

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### 4. Set up Firebase
1. Download your Firebase service account JSON file
2. Place it in `config/firebase-credentials/`
3. Update the path in your `.env` file

### 5. Run the Application
```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 🔧 Development

### Backend Development
The backend is built with FastAPI and organized into:
- **API Layer** (`backend/api/`): FastAPI routes and endpoints
- **Services Layer** (`backend/services/`): Business logic and external integrations
- **Models Layer** (`backend/models/`): Data models and schemas

### Frontend Development
The frontend is a single-page React application with:
- **Static Assets** (`frontend/static/`): Images, icons, and static files
- **Components** (`frontend/components/`): React components and pages
- **Main App** (`frontend/index.html`): Main React application

### Data Processing
Document processing scripts are in `data_processing/chunking/`:
- PDF chunking and text extraction
- Section classification
- Data preprocessing for RAG systems

## 🐳 Docker Deployment

### Build the Container
```bash
docker build -t icc-chatbot .
```

### Run the Container
```bash
docker run -p 8000:8000 \
  -e JWT_SECRET_KEY=your-secret-key \
  -e FIREBASE_SERVICE_ACCOUNT_PATH=/app/config/firebase-credentials/icc-project-472009-firebase-adminsdk.json \
  icc-chatbot
```

## ☁️ Cloud Run Deployment

### Prerequisites
- Google Cloud SDK installed
- Project configured with Cloud Run API enabled
- Firestore database set up

### Deploy to Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project-id/icc-chatbot

# Deploy to Cloud Run
gcloud run deploy icc-chatbot \
  --image gcr.io/your-project-id/icc-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars JWT_SECRET_KEY=your-secret-key
```

### Required Secrets
1. **JWT_SECRET_KEY**: Strong secret for JWT token signing
2. **Firebase Service Account**: JSON credentials for Firestore access

## 📊 Data Processing

### Document Chunking
Process PDF documents into searchable chunks:
```bash
cd data_processing/chunking
python run_geneva_chunking.py
```

### Notebook Analysis
Run Jupyter notebooks for data analysis:
```bash
cd data_processing/notebooks
jupyter notebook ICC_Enhanced_RAG_Production.ipynb
```

## 🧪 Testing

### Run Authentication Tests
```bash
cd backend
python test_auth.py
```

### Test API Endpoints
```bash
# Test login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass123"}'

# Test chat (requires authentication)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{"message": "Hello, ICC Assistant!"}'
```

## 🔒 Security Considerations

- JWT tokens are used for authentication
- Passwords are hashed with bcrypt
- CORS is configured for development (update for production)
- Environment variables are used for sensitive configuration
- Firebase security rules should be configured

## 📝 API Documentation

### Authentication Endpoints
- `POST /auth/signup` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Get current user info
- `PUT /auth/profile` - Update user profile
- `DELETE /auth/account` - Delete user account

### Conversation Endpoints
- `GET /conversations` - Get user conversations
- `POST /conversations` - Create new conversation
- `PUT /conversations/{id}` - Update conversation
- `DELETE /conversations/{id}` - Delete conversation
- `POST /conversations/cleanup` - Clean up empty conversations

### Chat Endpoints
- `POST /chat` - Send message to AI assistant

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the API documentation at `/docs` when running the application

## 🔄 Version History

- **v1.0.0** - Initial release with authentication, conversation management, and document processing
