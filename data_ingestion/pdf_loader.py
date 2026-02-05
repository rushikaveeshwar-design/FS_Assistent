from typing import Tuple, List
from langchain_community.document_loaders import PyPDFLoader
from data_ingestion.rule_ingestion import ingest_rulebook
from data_ingestion.engg_ingestion import ingest_engg_doc

def ingest_rulebook_pdf(*, pdf_path:str, competition: str,
                        year: int, section: str, domain: str,
                        source: str) -> Tuple[List[str], List[dict]]:
    """PDF -> text -> chunk_rule_text -> RuleMetadata"""

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    full_text = "\n".join(doc.page_content for doc in docs)

    return ingest_rulebook(text=full_text, competition=competition,
                           year=year, section=section, 
                           domain=domain, source=source)

def ingest_engg_pdf(*, pdf_path: str, domain: str,
                    topic: str, source: str) -> Tuple[List[str], List[dict]]:
    """PDF -> text -> chunk_engg_text -> EnggMetadata"""

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    full_text = "\n".join(doc.page_content for doc in docs)

    return ingest_engg_doc(text=full_text, domain=domain,
                           topic=topic, source=source)
