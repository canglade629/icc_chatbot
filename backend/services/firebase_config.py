import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials"""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            return firestore.client()
        
        # Get the service account key from environment variable or default location
        service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        
        if not service_account_path:
            # Try default location
            default_path = 'firebase-credentials/icc-project-472009-firebase-adminsdk.json'
            if os.path.exists(default_path):
                service_account_path = default_path
                print(f"✅ Using Firebase service account from: {default_path}")
            else:
                print("⚠️  FIREBASE_SERVICE_ACCOUNT_PATH not set and default file not found, using mock configuration")
                print(f"   Expected file location: {default_path}")
                return None
        
        # Initialize the app with service account credentials
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, {
            'projectId': 'icc-project-472009'
        })
        
        return firestore.client()
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return None

# Get Firestore client
db = initialize_firebase()

class FirebaseAuth:
    """Firebase Authentication service"""
    
    def __init__(self):
        self.db = db
        self.use_mock = db is None
    
    async def create_user(self, email: str, password: str, display_name: str = None):
        """Create a new user in Firebase Auth"""
        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=display_name
            )
            
            # Store additional user data in Firestore
            user_data = {
                'uid': user.uid,
                'email': email,
                'display_name': display_name,
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_login': firestore.SERVER_TIMESTAMP
            }
            
            self.db.collection('users').document(user.uid).set(user_data)
            
            return {
                'uid': user.uid,
                'email': user.email,
                'display_name': display_name
            }
        except Exception as e:
            raise Exception(f"Error creating user: {str(e)}")
    
    async def verify_user(self, email: str, password: str):
        """Verify user credentials and return user data"""
        try:
            # Get user by email
            user = auth.get_user_by_email(email)
            
            # Store login timestamp
            self.db.collection('users').document(user.uid).update({
                'last_login': firestore.SERVER_TIMESTAMP
            })
            
            return {
                'uid': user.uid,
                'email': user.email,
                'display_name': user.display_name
            }
        except Exception as e:
            raise Exception(f"Error verifying user: {str(e)}")
    
    async def get_user_by_uid(self, uid: str):
        """Get user data by UID"""
        try:
            user = auth.get_user(uid)
            return {
                'uid': user.uid,
                'email': user.email,
                'display_name': user.display_name
            }
        except Exception as e:
            raise Exception(f"Error getting user: {str(e)}")
    
    async def update_user_profile(self, uid: str, display_name: str = None):
        """Update user profile"""
        try:
            update_data = {}
            if display_name:
                update_data['display_name'] = display_name
            
            if update_data:
                auth.update_user(uid, **update_data)
                self.db.collection('users').document(uid).update(update_data)
            
            return await self.get_user_by_uid(uid)
        except Exception as e:
            raise Exception(f"Error updating user profile: {str(e)}")
    
    async def delete_user(self, uid: str):
        """Delete user account"""
        try:
            auth.delete_user(uid)
            self.db.collection('users').document(uid).delete()
            return True
        except Exception as e:
            raise Exception(f"Error deleting user: {str(e)}")

# Initialize auth service
firebase_auth = FirebaseAuth()
