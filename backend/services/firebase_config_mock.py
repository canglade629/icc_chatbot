"""
Mock Firebase configuration for testing without actual Firebase setup
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any

class MockFirebaseAuth:
    """Mock Firebase Authentication service for testing"""
    
    def __init__(self):
        self.users = {}  # In-memory user storage
        self.conversations = {}  # In-memory conversation storage
        self.next_uid = 1
    
    async def create_user(self, email: str, password: str, display_name: str = None):
        """Create a new user (mock)"""
        if email in [user['email'] for user in self.users.values()]:
            raise Exception("User already exists")
        
        uid = str(self.next_uid)
        self.next_uid += 1
        
        user_data = {
            'uid': uid,
            'email': email,
            'display_name': display_name,
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat()
        }
        
        self.users[uid] = user_data
        return user_data
    
    async def verify_user(self, email: str, password: str):
        """Verify user credentials (mock)"""
        for user in self.users.values():
            if user['email'] == email:
                # In mock, any password works
                user['last_login'] = datetime.now().isoformat()
                return user
        
        raise Exception("Invalid credentials")
    
    async def get_user_by_uid(self, uid: str):
        """Get user data by UID (mock)"""
        if uid not in self.users:
            raise Exception("User not found")
        return self.users[uid]
    
    async def update_user_profile(self, uid: str, display_name: str = None):
        """Update user profile (mock)"""
        if uid not in self.users:
            raise Exception("User not found")
        
        if display_name:
            self.users[uid]['display_name'] = display_name
        
        return self.users[uid]
    
    async def delete_user(self, uid: str):
        """Delete user account (mock)"""
        if uid not in self.users:
            raise Exception("User not found")
        
        del self.users[uid]
        return True

    # Conversation management methods
    async def create_conversation(self, uid: str, title: str = "New Conversation"):
        """Create a new conversation for a user (mock)"""
        conversation_id = f"conv_{len(self.conversations) + 1}"
        conversation = {
            "id": conversation_id,
            "uid": uid,
            "title": title,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.conversations[conversation_id] = conversation
        return conversation

    async def get_user_conversations(self, uid: str):
        """Get all conversations for a user (mock)"""
        return [conv for conv in self.conversations.values() if conv["uid"] == uid]

    async def update_conversation(self, conversation_id: str, title: str = None, messages: list = None):
        """Update a conversation (mock)"""
        if conversation_id in self.conversations:
            if title is not None:
                self.conversations[conversation_id]["title"] = title
            if messages is not None:
                self.conversations[conversation_id]["messages"] = messages
            self.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            
            # Enforce conversation limit after update
            uid = self.conversations[conversation_id]["uid"]
            await self._enforce_conversation_limit(uid)
            
            return True
        return False

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation (mock)"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    async def _enforce_conversation_limit(self, uid: str, max_conversations: int = 20):
        """Enforce conversation limit by deleting oldest conversations (mock)"""
        try:
            user_conversations = [conv for conv in self.conversations.values() if conv["uid"] == uid]
            
            # If we have more than the limit, delete the oldest ones
            if len(user_conversations) >= max_conversations:
                # Sort by updated_at (oldest first)
                user_conversations.sort(key=lambda x: x['updated_at'])
                
                # Delete the oldest conversations
                conversations_to_delete = user_conversations[:len(user_conversations) - max_conversations + 1]
                for conv in conversations_to_delete:
                    if conv['id'] in self.conversations:
                        del self.conversations[conv['id']]
                
                print(f"Mock: Deleted {len(conversations_to_delete)} old conversations for user {uid}")
                
        except Exception as e:
            print(f"Mock: Error enforcing conversation limit: {e}")

# Mock Firestore client
class MockFirestoreClient:
    def collection(self, name):
        return MockCollection()

class MockCollection:
    def document(self, doc_id):
        return MockDocument()

class MockDocument:
    def set(self, data):
        pass
    
    def update(self, data):
        pass
    
    def delete(self):
        pass

# Initialize mock services
firebase_auth = MockFirebaseAuth()
db = MockFirestoreClient()
