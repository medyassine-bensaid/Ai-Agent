from typing import List, Dict
from datetime import datetime

class ConversationMemory:
    """Simple in-memory conversation storage"""
    
    def __init__(self, max_messages: int = 10):
        self.messages: List[Dict] = []
        self.max_messages = max_messages
        self.metadata = {
            "user_expertise": "unknown",
            "topics_discussed": set(),
            "user_stack": set()
        }
    
    def add_message(self, role: str, content: str):
        """Add a message to memory"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last N messages
        if len(self.messages) > self.max_messages * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_messages * 2:]
    
    def get_context(self) -> str:
        """Get conversation context as string"""
        if not self.messages:
            return "No previous conversation."
        
        context = "Recent conversation:\n"
        for msg in self.messages[-6:]:  # Last 3 exchanges
            role = msg["role"].upper()
            content = msg["content"][:200]  # Truncate long messages
            context += f"{role}: {content}\n"
        
        return context
    
    def update_metadata(self, key: str, value):
        """Update conversation metadata"""
        if key in ["topics_discussed", "user_stack"]:
            self.metadata[key].add(value)
        else:
            self.metadata[key] = value
    
    def get_metadata_context(self) -> str:
        """Get metadata as context string"""
        topics = ", ".join(list(self.metadata["topics_discussed"])[:5])
        stack = ", ".join(list(self.metadata["user_stack"])[:5])
        
        context = f"User expertise: {self.metadata['user_expertise']}\n"
        if topics:
            context += f"Topics discussed: {topics}\n"
        if stack:
            context += f"User's tech stack: {stack}\n"
        
        return context
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.metadata = {
            "user_expertise": "unknown",
            "topics_discussed": set(),
            "user_stack": set()
        }

# Global memory instance
conversation_memory = ConversationMemory()