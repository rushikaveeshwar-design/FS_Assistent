from data_ingestion.chunking import chunk_engg_text
from data_ingestion.metadata import EnggMetadata

def ingest_engg_doc(text: str, domain: str, 
                    topic: str, source: str):
    chunks = chunk_engg_text(text)
    metadatas = [EnggMetadata(domain=domain, topic=topic,
                              source=source).__dict__ for _ in chunks]
    
    return chunks, metadatas
