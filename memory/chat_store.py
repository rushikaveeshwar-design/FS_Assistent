import sqlite3
from datetime import datetime, timezone

class ChatStore:
    def __init__(self, db_path="chats.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self.init_document_tables()

    def _create_tables(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chats(
                            chat_id TEXT PRIMARY KEY,
                            title TEXT, created_at TEXT,
                            last_active TEXT)""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS messages(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            chat_id TEXT, role TEXT,
                            content TEXT, timestamp TEXT)""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chat_memory(
                          chat_id TEXT PRIMARY KEY,
                          memory TEXT,
                          updated_at TEXT)""")
            
    def add_messages(self, chat_id, role, content):
        self.conn.execute("INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)",
                          (chat_id, role, content, datetime.now(timezone.utc).isoformat()))
        
        self.conn.commit()
    
    def get_messages(self, chat_id):
        cursor = self.conn.execute("""SELECT role, 
                                   content FROM messages 
                                   WHERE chat_id = ? 
                                   ORDER BY id ASC""",
                                   (chat_id,))
        return [{"role":role, "content":content} for role, content in cursor.fetchall()]
    
    def create_chat(self, chat_id, title):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""INSERT OR IGNORE INTO chats
                          (chat_id, title, created_at, last_active)
                          VALUES (?, ?, ?, ?)""",
                          (chat_id, title, now, now))
        self.conn.commit()
    
    def list_chats(self):
        cursor = self.conn.execute("""SELECT chat_id, title
                                   FROM chats
                                   ORDER BY last_active DESC""")
        return cursor.fetchall()
    
    def touch_chat(self, chat_id):
        self.conn.execute("""UPDATE chats
                          SET last_active = ?
                          WHERE chat_id = ?""",
                          (datetime.now(timezone.utc).isoformat(), chat_id))
        self.conn.commit()
    
    def search_messages(self, chat_id, query: str):
        cursor = self.conn.execute("""SELECT role, content
                                   FROM messages
                                   WHERE chat_id = ?
                                   AND content LIKE ?
                                   ORDER BY id ASC""",(chat_id, f"%{query}%"))
        return [{"role":role, "content": content} for role, content in cursor.fetchall()]
    
    def init_document_tables(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS documents 
                          (document_id TEXT PRIMARY KEY,
                          doc_type TEXT, vectorstore_path TEXT,
                          created_at TEXT)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chat_documents
                          (chat_id TEXT, document_id TEXT,
                          PRIMARY KEY (chat_id, document_id))""")
        self.conn.commit()
    
    def attach_document_to_chat(self, chat_id: str, document_id: str, conn):
        self.conn.execute("""
                    INSERT OR IGNORE INTO chat_documents (chat_id, document_id)
                    VALUES (?, ?)""", (chat_id, document_id))
        self.conn.commit()

    def document_exists(self, document_id: str):
        cursor = self.conn.execute("SELECT 1 FROM documents WHERE document_id = ?",
                                   (document_id,))
        return cursor.fetchone() is not None
    
    def get_memory(self, chat_id):
        cur = self.conn.execute("SELECT memory FROM chat_memory WHERE chat_id=?",
        (chat_id,))
        row = cur.fetchone()
        return row[0] if row else ""
    
    def update_memory(self, chat_id, summary):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""INSERT INTO chat_memory(chat_id, memory, updated_at)
                          VALUES (?,?,?)
                          ON CONFLICT(chat_id)
                          DO UPDATE SET memory=excluded.memory,
                          updated_at=excluded.updated_at""", (chat_id, summary, now))
        self.conn.commit()

def generate_chat_title(question: str, mode: str):
    return f"{mode}: {question[:40]}"
