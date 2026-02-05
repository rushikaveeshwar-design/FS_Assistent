import hashlib
import yaml

def compute_pdf_sha256(pdf_path: str):
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_document_registry(yaml_path="config\documents.yaml") -> dict:
    with open(yaml_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

def resolve_rulebook_documents(registry, competition, year):
    docs = []
    for doc in registry["rulebooks"]:
        if doc["competition"] != competition:
            continue
        if doc["year"] != year:
            continue
        docs.append(doc)
    return docs

def resolve_engineering_documents(registry: dict, domain: str,
                                  topic: str):
    """Resolve relevant engineering documents from registry
    based on inferred domain and/or topic.

    Parameters:-

    registry : dict
        Loaded document registry (e.g. from documents.yaml)
    domain : str | None
        Engineering domain (e.g. 'chassis', 'powertrain')
    topic : str | None
        Optional finer-grained topic (e.g. 'vehicle_dynamics')"""
    
    docs = []
    
    for doc in registry["engineering"]:
        if doc["domain"] != domain:
            continue
        if doc["topic"] != topic:
            continue

        docs.append(doc)
    
    return docs
