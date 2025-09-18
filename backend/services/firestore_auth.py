"""
Firestore-only authentication system
Uses Firestore to store user data and passwords with bcrypt hashing
"""
import firebase_admin
from firebase_admin import credentials, firestore
from passlib.context import CryptContext
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Password hashing
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
except Exception as e:
    print(f"Warning: bcrypt context creation failed: {e}")
    # Fallback to a simpler configuration
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class FirestoreAuth:
    """Firestore-based authentication service"""
    
    def __init__(self):
        self.db = self._initialize_firestore()
        self.pwd_context = pwd_context
        self.use_mock = self.db is None
    
    def _initialize_firestore(self):
        """Initialize Firestore connection"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                return firestore.client()
            
            # For Cloud Run, try to use default credentials first
            try:
                # Try to initialize with default credentials (works in Cloud Run)
                firebase_admin.initialize_app(options={
                    'projectId': 'icc-project-472009'
                })
                print("✅ Firebase initialized with default credentials")
                return firestore.client()
            except Exception as default_error:
                print(f"Default credentials failed: {default_error}")
                
                # Fallback to service account file
                service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
                if not service_account_path:
                    # Try default locations
                    possible_paths = [
                        'config/firebase-credentials/icc-project-472009-firebase-adminsdk.json',
                        'firebase-credentials/icc-project-472009-firebase-adminsdk.json',
                        '../../config/firebase-credentials/icc-project-472009-firebase-adminsdk.json'
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            service_account_path = path
                            break
                
                if not service_account_path:
                    raise ValueError("Firebase service account not found in any expected location")
                
                # Initialize the app with service account credentials
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': 'icc-project-472009'
                })
                print("✅ Firebase initialized with service account file")
                return firestore.client()
        except Exception as e:
            print(f"❌ Error initializing Firestore: {e}")
            print("🔄 Falling back to mock authentication")
            return None
    
    def _hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def create_user(self, email: str, password: str, display_name: str = None):
        """Create a new user in Firestore"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.create_user(email, password, display_name)
        
        try:
            # Check if user already exists
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).limit(1)
            existing_users = query.get()
            
            if existing_users:
                raise Exception("User with this email already exists")
            
            # Create user document
            user_id = f"user_{int(datetime.now().timestamp() * 1000)}"
            hashed_password = self._hash_password(password)
            
            user_data = {
                'uid': user_id,
                'email': email,
                'display_name': display_name,
                'password_hash': hashed_password,
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_login': firestore.SERVER_TIMESTAMP,
                'is_active': True
            }
            
            # Store user in Firestore
            self.db.collection('users').document(user_id).set(user_data)
            
            return {
                'uid': user_id,
                'email': email,
                'display_name': display_name
            }
        except Exception as e:
            raise Exception(f"Error creating user: {str(e)}")
    
    async def verify_user(self, email: str, password: str):
        """Verify user credentials"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.verify_user(email, password)
        
        try:
            # Find user by email
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).limit(1)
            users = query.get()
            
            if not users:
                raise Exception("Invalid credentials")
            
            user_doc = users[0]
            user_data = user_doc.to_dict()
            
            # Check if user is active
            if not user_data.get('is_active', True):
                raise Exception("Account is deactivated")
            
            # Verify password
            if not self._verify_password(password, user_data['password_hash']):
                raise Exception("Invalid credentials")
            
            # Update last login
            user_doc.reference.update({
                'last_login': firestore.SERVER_TIMESTAMP
            })
            
            return {
                'uid': user_data['uid'],
                'email': user_data['email'],
                'display_name': user_data.get('display_name')
            }
        except Exception as e:
            raise Exception(f"Error verifying user: {str(e)}")
    
    async def get_user_by_uid(self, uid: str):
        """Get user data by UID"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.get_user_by_uid(uid)
        
        try:
            user_doc = self.db.collection('users').document(uid).get()
            
            if not user_doc.exists:
                raise Exception("User not found")
            
            user_data = user_doc.to_dict()
            
            return {
                'uid': user_data['uid'],
                'email': user_data['email'],
                'display_name': user_data.get('display_name')
            }
        except Exception as e:
            raise Exception(f"Error getting user: {str(e)}")
    
    async def update_user_profile(self, uid: str, display_name: str = None, email: str = None):
        """Update user profile"""
        try:
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                raise Exception("User not found")
            
            update_data = {}
            if display_name is not None:
                update_data['display_name'] = display_name
            if email is not None:
                # Check if email is already taken
                users_ref = self.db.collection('users')
                query = users_ref.where('email', '==', email).limit(1)
                existing_users = query.get()
                if existing_users and existing_users[0].id != uid:
                    raise Exception("Email already in use")
                update_data['email'] = email
            
            if update_data:
                user_ref.update(update_data)
            
            return await self.get_user_by_uid(uid)
        except Exception as e:
            raise Exception(f"Error updating user profile: {str(e)}")
    
    async def change_password(self, uid: str, old_password: str, new_password: str):
        """Change user password"""
        try:
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                raise Exception("User not found")
            
            user_data = user_doc.to_dict()
            
            # Verify old password
            if not self._verify_password(old_password, user_data['password_hash']):
                raise Exception("Current password is incorrect")
            
            # Update password
            new_password_hash = self._hash_password(new_password)
            user_ref.update({
                'password_hash': new_password_hash
            })
            
            return True
        except Exception as e:
            raise Exception(f"Error changing password: {str(e)}")
    
    async def delete_user(self, uid: str):
        """Delete user account"""
        try:
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                raise Exception("User not found")
            
            # Soft delete - mark as inactive
            user_ref.update({
                'is_active': False,
                'deleted_at': firestore.SERVER_TIMESTAMP
            })
            
            return True
        except Exception as e:
            raise Exception(f"Error deleting user: {str(e)}")

    # Conversation management methods
    async def create_conversation(self, uid: str, title: str = "New Conversation"):
        """Create a new conversation for a user"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.create_conversation(uid, title)
        
        try:
            # Check current conversation count and clean up if needed
            await self._enforce_conversation_limit(uid)
            
            conversation_ref = self.db.collection('conversations').document()
            now = datetime.utcnow()
            conversation_data = {
                'uid': uid,
                'title': title,
                'messages': [],
                'created_at': now,
                'updated_at': now
            }
            conversation_ref.set(conversation_data)
            
            return {
                'id': conversation_ref.id,
                'uid': uid,
                'title': title,
                'messages': [],
                'created_at': now.isoformat(),
                'updated_at': now.isoformat()
            }
        except Exception as e:
            raise Exception(f"Error creating conversation: {str(e)}")

    async def get_user_conversations(self, uid: str):
        """Get all conversations for a user"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.get_user_conversations(uid)
        
        try:
            conversations_ref = self.db.collection('conversations')
            query = conversations_ref.where('uid', '==', uid)
            docs = query.stream()
            
            conversations = []
            for doc in docs:
                conversation_data = doc.to_dict()
                
                # Convert timestamps to ISO format strings
                created_at = conversation_data.get('created_at')
                updated_at = conversation_data.get('updated_at')
                
                if hasattr(created_at, 'timestamp'):
                    created_at = created_at.timestamp()
                if hasattr(updated_at, 'timestamp'):
                    updated_at = updated_at.timestamp()
                
                # Convert to ISO string if it's a timestamp
                if isinstance(created_at, (int, float)):
                    created_at = datetime.fromtimestamp(created_at).isoformat()
                elif isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                else:
                    created_at = str(created_at) if created_at else datetime.utcnow().isoformat()
                
                if isinstance(updated_at, (int, float)):
                    updated_at = datetime.fromtimestamp(updated_at).isoformat()
                elif isinstance(updated_at, datetime):
                    updated_at = updated_at.isoformat()
                else:
                    updated_at = str(updated_at) if updated_at else datetime.utcnow().isoformat()
                
                conversations.append({
                    'id': doc.id,
                    'uid': conversation_data['uid'],
                    'title': conversation_data['title'],
                    'messages': conversation_data.get('messages', []),
                    'created_at': created_at,
                    'updated_at': updated_at
                })
            return conversations
        except Exception as e:
            raise Exception(f"Error getting conversations: {str(e)}")

    async def update_conversation(self, conversation_id: str, title: str = None, messages: list = None):
        """Update a conversation"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.update_conversation(conversation_id, title, messages)
        
        try:
            conversation_ref = self.db.collection('conversations').document(conversation_id)
            update_data = {'updated_at': datetime.utcnow()}
            
            if title is not None:
                update_data['title'] = title
            if messages is not None:
                update_data['messages'] = messages
            
            conversation_ref.update(update_data)
            
            # Get the user ID from the conversation to enforce limits
            conversation_doc = conversation_ref.get()
            if conversation_doc.exists:
                conversation_data = conversation_doc.to_dict()
                uid = conversation_data.get('uid')
                if uid:
                    await self._enforce_conversation_limit(uid)
            
            return True
        except Exception as e:
            raise Exception(f"Error updating conversation: {str(e)}")

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth.delete_conversation(conversation_id)
        
        try:
            self.db.collection('conversations').document(conversation_id).delete()
            return True
        except Exception as e:
            raise Exception(f"Error deleting conversation: {str(e)}")

    async def _enforce_conversation_limit(self, uid: str, max_conversations: int = 20):
        """Enforce conversation limit by deleting oldest conversations"""
        if self.use_mock:
            # Mock implementation for testing
            from firebase_config_mock import firebase_auth as mock_auth
            return await mock_auth._enforce_conversation_limit(uid, max_conversations)
        
        try:
            conversations_ref = self.db.collection('conversations')
            query = conversations_ref.where('uid', '==', uid)
            docs = query.stream()
            
            conversations = []
            for doc in docs:
                conversation_data = doc.to_dict()
                conversations.append({
                    'id': doc.id,
                    'created_at': conversation_data.get('created_at'),
                    'updated_at': conversation_data.get('updated_at')
                })
            
            # If we have more than the limit, delete the oldest ones
            if len(conversations) >= max_conversations:
                # Sort by updated_at (oldest first)
                conversations.sort(key=lambda x: x['updated_at'] if x['updated_at'] else x['created_at'])
                
                # Delete the oldest conversations
                conversations_to_delete = conversations[:len(conversations) - max_conversations + 1]
                for conv in conversations_to_delete:
                    await self.delete_conversation(conv['id'])
                
                print(f"Deleted {len(conversations_to_delete)} old conversations for user {uid}")
                
        except Exception as e:
            print(f"Error enforcing conversation limit: {e}")

# Initialize auth service
firestore_auth = FirestoreAuth()
