from langchain_community.vectorstores import FAISS
from langchain.embeddings import Embeddings
from typing_extensions import List

RULE_STORE_PATH = "vectorstores/rules_text"
ENGG_STORE_PATH = "vectorstores/engg_text"
IMAGE_STORE_PATH = "vectorstores/images"

class VectorStoreManager:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
    
    def create_store(self, texts: List[str], 
                     metadata: List[dict]):
        return FAISS.from_texts(texts=texts, 
                                embedding=self.embedding_model,
                                metadatas=metadata)
    
    def create_store_from_images(self, image_embeddings: List[List[float]],
                                 meatadata: List[dict]):
        """Used for CLIP image embeddings."""
        return FAISS.from_embeddings(embeddings=image_embeddings,
                                     metadatas=meatadata)
    
    def load_store(self, path: str):
        return FAISS.load_local(path, self.embedding_model)
    
    def save_store(self, store, path: str):
        store.save_local(path)
    

def load_rule_store(embedding_model: Embeddings):
    manager = VectorStoreManager(embedding_model)
    return manager.load_store(RULE_STORE_PATH)

def load_engg_store(embedding_model: Embeddings):
    manager = VectorStoreManager(embedding_model)
    return manager.load_store(ENGG_STORE_PATH)

def load_image_store(embedding_model: Embeddings):
    manager = VectorStoreManager(embedding_model)
    return manager.load_store(IMAGE_STORE_PATH)