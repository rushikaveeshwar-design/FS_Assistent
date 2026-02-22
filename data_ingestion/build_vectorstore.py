import os
import time
from datetime import datetime, timezone
from retrieval.vectorstore import VectorStoreManager
from data_ingestion.clip_embedder import embed_images
from retrieval.vectorstore import (RULE_STORE_PATH, ENGG_STORE_PATH, 
                                   IMAGE_STORE_PATH)
from data_ingestion.doc_utils import compute_pdf_sha256
from data_ingestion.pdf_loader import ingest_rulebook_pdf, ingest_engg_pdf
from data_ingestion.pdf_image_extractor import extract_images_from_pdf
from agent.logger import log_event, log_exception

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

def get_or_create_global_vectorstore(*, document_id: str,
                                     doc_type: str, ingest_fn,
                                     vectorstore_manager, base_path: str,
                                     chat_store, chat_id):
    """Return a vectorstore for a document.
        Ingests exactly once globally using SHA-256 document_id.
    """
    store_path = os.path.join(base_path, document_id)

    if os.path.exists(store_path):
        log_event("INFO", "vectorstore_cache_hit",
                  chat_id=chat_id, meta={"doc_type": doc_type,
                                         "document_id": document_id,
                                         "path":store_path})
        chat_store.attach_document_to_chat(chat_id, document_id)
        return vectorstore_manager.load_store(store_path)
    
    # cache miss (ingest)
    try:
        t0 = time.time()

        log_event("INFO", "vectorstore_cache_miss",
                  chat_id=chat_id, meta={"doc_type": doc_type,
                                         "document_id": document_id})

        texts, metadatas = ingest_fn()

        t_ingest = int((time.time() - t0)*1000)

        store = vectorstore_manager.create_store(texts, metadatas)
        vectorstore_manager.save_store(store, store_path)

        # Register document globally
        chat_store.conn.execute("""INSERT OR IGNORE INTO documents
                    (document_id, doc_type, vectorstore_path, created_at)
                    VALUES (?, ?, ?, ?)""", (document_id, doc_type,
                                            store_path, datetime.now(timezone.utc).isoformat()))
        
        chat_store.conn.commit()
        chat_store.attach_document_to_chat(chat_id, document_id)

        log_event("INFO", "vectorstore_ingest_complete",
                chat_id=chat_id, meta={"doc_type":doc_type,
                                        "document_id": document_id,
                                        "chunks": len(texts),
                                        "latency": t_ingest,
                                        "path": store_path})

        return store
    
    except Exception as e:
        log_exception(e, chat_id=chat_id, node="vectorstore_ingest")
        raise

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