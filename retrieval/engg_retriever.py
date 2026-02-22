import time
from agent.logger import log_event, log_exception

def retrieve_engg(store, query: str, k: int=3, chat_id=None):

    t0 = time.time()

    try:
        docs = store.similarity_search(query, k=k)
        latency = int((time.time() - t0)*1000)

        log_event("DEBUG", "vectorstore_query_time",
                  chat_id=chat_id,
                  meta={"store_type": "engineering",
                        "results": len(docs), "latency": latency})
        return docs
    
    except Exception as e:
        log_exception(e, node="retrieve_engg")
        raise
