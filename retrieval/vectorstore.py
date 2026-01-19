from langchain_community.vectorstores import FAISS
from langchain.embeddings import Embeddings

class VectorStoreManager:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
    
    def create_store(self, texts, metadatas):
        return FAISS.from_texts(texts=texts, 
                                embedding=self.embedding_model,
                                metadatas=metadatas)
    
    def load_store(self, path: str):
        return FAISS.load_local(path, self.embedding_model)
    
    def save_store(self, store, path: str):
        store.save_local(path)
    
    