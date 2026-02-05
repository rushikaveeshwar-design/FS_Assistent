import os
from datetime import datetime, timezone
from retrieval.vectorstore import VectorStoreManager
from data_ingestion.clip_embedder import embed_images
from retrieval.vectorstore import (RULE_STORE_PATH, ENGG_STORE_PATH, 
                                   IMAGE_STORE_PATH)
from data_ingestion.doc_utils import compute_pdf_sha256
from data_ingestion.pdf_loader import ingest_rulebook_pdf, ingest_engg_pdf
from data_ingestion.pdf_image_extractor import extract_images_from_pdf

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

def get_or_create_global_vectorstore(*, document_id: str,
                                     doc_type: str, ingest_fn,
                                     vectorstore_manager, base_path: str,
                                     chat_store, chat_id):
    """Return a vectorstore for a document.
        Ingests exactly once globally using SHA-256 document_id.
    """
    store_path = os.path.join(base_path, document_id)

    if os.path.exists(store_path):
        chat_store.attach_document_to_chat(chat_id, document_id)
        return vectorstore_manager.load_store(store_path)
    
    texts, metadatas = ingest_fn()
    store = vectorstore_manager.create_store(texts, metadatas)
    vectorstore_manager.save_store(store, store_path)

    # Register document globally
    chat_store.conn.execute("""INSERT OR IGNORE INTO documents
                 (document_id, doc_type, vectorstore_path, create_at)
                 VALUES (?, ?, ?, ?)""", (document_id, doc_type,
                                          store_path, datetime.now(timezone.utc).isoformat()))
    
    chat_store.conn.commit()
    chat_store.attach_document_to_chat(chat_id, document_id)

    return store

def load_rule_vectorstores(descriptors, chat_id, 
                           chat_store, vectorstore_manager):
    stores = []

    for doc in descriptors:
        document_id = compute_pdf_sha256(doc["pdf_path"])

        store = get_or_create_global_vectorstore(document_id=document_id, doc_type="rulebook",
                                                 ingest_fn=lambda d=doc: ingest_rulebook_pdf(pdf_path=d["pdf_path"],
                                                                                             competition=d["competition"],
                                                                                             year=d["year"],
                                                                                             section=",".join(d["sections"]),
                                                                                             domain=None,source=d["source"]),
                                                vectorstore_manager=vectorstore_manager, base_path="vstores\rules",
                                                chat_store=chat_store, chat_id=chat_id)
        stores.append(store)
    return stores

def load_engg_vectorstores(descriptors, chat_id, chat_store, vectorstore_manager):
    stores = []

    for doc in descriptors:
        document_id = compute_pdf_sha256(doc["pdf_path"])

        store = get_or_create_global_vectorstore(document_id=document_id,
                                                 doc_type="engineering",
                                                 ingest_fn=lambda d=doc: ingest_engg_pdf(pdf_path=d["pdf_path"],
                                                                                         domain=d["domain"],
                                                                                         topic=d["topic"],
                                                                                         source=d["source"]),
                                                vectorstore_manager=vectorstore_manager, base_path="vstores\engg", 
                                                chat_store=chat_store, chat_id=chat_id)
        stores.append(store)
    
    return stores

def load_image_vectorstores(descriptors, chat_id,
                            chat_store, vectorstore_manager,
                            clip_model, preprocess, device):
    stores = []

    for doc in descriptors:
        document_id = compute_pdf_sha256(doc["pdf_path"])

        def ingest_fn(d=doc):
            extracted = extract_images_from_pdf(d["pdf_path"])
            images = [img for img, _ in extracted]

            metadatas = []
            for _, metadata in extracted:
                metadatas.append({"source": d["source"],
                                  "page": metadata["page"],
                                  "competition": d["competition"],
                                  "year": d["year"], "domain": d["domain"],
                                  "section": d["section"]})
            
            embeddings = embed_images(images, clip_model=clip_model,
                                      preprocess=preprocess, device=device)
            return embeddings, metadatas
        
        store = get_or_create_global_vectorstore(document_id=document_id,
                                                 doc_type="images", ingest_fn=ingest_fn,
                                                 vectorstore_manager=vectorstore_manager,
                                                 base_path="vstores/images",
                                                 chat_store=chat_store, chat_id=chat_id)
        
        stores.append(store)
    return stores