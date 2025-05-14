import sqlite3
import os
import json
import uuid
import datetime
from typing import List, Dict, Optional, Tuple, Any

# Database configuration
DB_PATH = "chat_history.db"

def init_db():
    """Initialize the database with necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        content TEXT NOT NULL,
        role TEXT NOT NULL,
        logs TEXT,
        tools_used TEXT,
        timestamp TIMESTAMP NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {DB_PATH}")

def create_conversation(title: str = "New Conversation") -> str:
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    conversation_id = str(uuid.uuid4())
    now = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conversation_id, title, now, now)
    )
    
    conn.commit()
    conn.close()
    
    return conversation_id

def add_message(
    conversation_id: str,
    content: str,
    role: str,
    logs: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None
) -> str:
    """
    Add a message to a conversation and return the message ID.
    
    Args:
        conversation_id: The ID of the conversation
        content: The message content
        role: Either 'user' or 'agent'
        logs: Optional list of log messages
        tools_used: Optional list of tool names used
    
    Returns:
        The ID of the created message
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert logs and tools to JSON strings
    logs_json = json.dumps(logs) if logs else None
    tools_used_json = json.dumps(tools_used) if tools_used else None
    
    message_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO messages (id, conversation_id, content, role, logs, tools_used, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (message_id, conversation_id, content, role, logs_json, tools_used_json, timestamp)
    )
    
    # Update the conversation's updated_at timestamp
    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (timestamp, conversation_id)
    )
    
    conn.commit()
    conn.close()
    
    return message_id

def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID with all its messages."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Get conversation details
    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    conversation = cursor.fetchone()
    
    if not conversation:
        conn.close()
        return None
    
    # Convert to dict
    conversation_dict = dict(conversation)
    
    # Get messages for this conversation
    cursor.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", 
        (conversation_id,)
    )
    messages = [dict(row) for row in cursor.fetchall()]
    
    # Process each message to parse JSON fields
    for message in messages:
        if message['logs']:
            message['logs'] = json.loads(message['logs'])
        if message['tools_used']:
            message['tools_used'] = json.loads(message['tools_used'])
    
    conversation_dict['messages'] = messages
    
    conn.close()
    return conversation_dict

def get_all_conversations() -> List[Dict[str, Any]]:
    """Get all conversations with their metadata (no messages)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.*, COUNT(m.id) as message_count 
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """)
    
    conversations = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return conversations

def update_conversation_title(conversation_id: str, new_title: str) -> bool:
    """Update a conversation's title."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (new_title, conversation_id)
    )
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success

def extract_title_from_first_message(conversation_id: str) -> Optional[str]:
    """Extract a title from the first message in a conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT 1", 
        (conversation_id,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    # Extract a title from the first few words (max 5 words and 50 characters)
    content = result[0]
    words = content.split()
    title = " ".join(words[:5])
    if len(title) > 50:
        title = title[:47] + "..."
        
    return title

# Initialize the database when the module is imported
init_db() 