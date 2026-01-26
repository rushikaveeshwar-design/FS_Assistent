import sqlite3
from datetime import datetime

class ChatStore:
    def __init__(self, db_path="chats.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chats(
                            chat_id TEXT PRIMARY KEY,
                            title TEXT, created_at TEXT,
                            last_active TEXT)""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS messages(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            chat_id TEXT, role TEXT,
                            content TEXT, timestamp TEXT)""")
            
    def add_messages(self, chat_id, role, content):
        self.conn.execute("INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)",
                          (chat_id, role, content, datetime.now(datetime.timezone.utc).isoformat()))
        
        self.conn.commit()

def generate_chat_title(question: str, mode: str):
    return f"{mode}: {question[:40]}"
