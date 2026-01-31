import os
from retrieval.vectorstore import VectorStoreManager
from data_ingestion.clip_embedder import embed_images
from retrieval.vectorstore import (RULE_STORE_PATH, ENGG_STORE_PATH, 
                                   IMAGE_STORE_PATH)

def build_rule_store(chunks, metadatas, embedding_model):
    os.makedirs(RULE_STORE_PATH, exist_ok=True)

    manager = VectorStoreManager(embedding_model)
    store = manager.create_store(chunks, metadatas)
    manager.save_store(store, RULE_STORE_PATH)


def build_engg_store(chunks, metadatas, embedding_model):
    os.makedirs(ENGG_STORE_PATH, exist_ok=True)

    manager = VectorStoreManager(embedding_model)
    store = manager.create_store(chunks, metadatas)
    manager.save_store(store, ENGG_STORE_PATH)


def build_image_store(images, metadatas, 
                      clip_model, preprocess, 
                      device, embedding_model):
    os.makedirs(IMAGE_STORE_PATH, exist_ok=True)

    embeddings = embed_images(images, clip_model, 
                              preprocess, device)

    manager = VectorStoreManager(embedding_model)
    store = manager.create_store_from_images(image_embedding=embeddings, 
                                             metadatas=metadatas)
    manager.save_store(store, IMAGE_STORE_PATH)

def get_or_create_chat_vectorstore(*, chat_id: str,
                                   document_id: str,
                                   ingest_fn, load_fn,
                                   save_fn, base_path: str):
    """Ensure vectorstore for (chat_id, document_id) exists.
    Ingest only once per conversation.
    """
    chat_path = os.path.join(base_path, chat_id)
    os.makedirs(chat_id, exist_ok=True)

    store_path = os.path.join(chat_path, document_id)

    if os.path.exists(store_path):
        return load_fn(store_path)
    
    texts, metadatas = ingest_fn()
    store = VectorStoreManager(...).create_store(texts, metadatas)
    save_fn(store, store_path)
    return store