import os
from retrieval.vectorstore import VectorStoreManager

RULE_STORE_PATH = "vectorstores/rules_text"
ENGG_STORE_PATH = "vectorstores/engg_text"
IMAGE_STORE_PATH = "vectorstores/images"

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


def build_image_store(image_embeddings, metadatas, embedding_model):
    os.makedirs(IMAGE_STORE_PATH, exist_ok=True)

    manager = VectorStoreManager(embedding_model)
    store = manager.create_store_from_images(image_embeddings, metadatas)
    manager.save_store(store, IMAGE_STORE_PATH)