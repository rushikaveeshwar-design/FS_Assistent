import time
from agent.logger import log_event, log_exception

def retrieve_rules(store, query: str, 
                   competition: str, year: int, 
                   domain: str | None=None, k: int=3, chat_id=None):
    
    t0 = time.time()

    try:


        filters = {"competition": competition,
                "year": year}
        if domain:
            filters["domain"] = domain

        docs = store.similarity_search(query, k=k,
                                   filter=filters) # Rule must be filtered before being trusted

        latency = int((time.time() - t0)*1000)

        log_event("DEBUG", "vectorstore_query_time",
                  chat_id=chat_id,
                  meta={"store_type": "rules",
                        "results": len(docs),
                        "latency": latency,
                        "competition": competition,
                        "year": year, "domain": domain})
        return docs
    
    except Exception as e:
        log_exception(e, node="retrieve_rules")
        raise