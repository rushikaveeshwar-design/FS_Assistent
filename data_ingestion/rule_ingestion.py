# from pathlib import Path
from data_ingestion.chunking import chunk_rule_text
from data_ingestion.metadata import RuleMetadata

def ingest_rulebook(text: str, competition: str, year: int,
                    section: str, domain: str, source: str):
    chunks = chunk_rule_text(text)
    metadatas = [RuleMetadata(competition=competition,
                             year=year, section=section,
                             domain=domain, source=source).__dict__ for _ in chunks]
    
    return chunks, metadatas
