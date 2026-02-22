import time
from data_ingestion.clip_embedder import embed_text_for_clip
from agent.logger import log_vision_event, log_event, log_exception

Image_Trigger = ["diagram", "figure", "layout", "image"]

def retrieve_relevant_images(query: str, tools, clip_model,
                             device, k: int=1, chat_id=None):
    
    t0 = time.time()

    try:
        query_vector = embed_text_for_clip(query, model=clip_model,
                                        device=device)
        
        docs = tools.get_image_by_vector(query_vector, k=k)

        results = []
        for d in docs:
            metadata = d.metadata
            results.append({"source": metadata["source"],
                            "page": metadata["page"],
                            "domain": metadata["domain"],
                            "competition": metadata["competition"],
                            "year": metadata["year"],
                            "section": metadata["section"]})
            
        latency = int((time.time() - t0)*1000)

        log_event("DEBUG", "vectorstore_query_time",
                chat_id=chat_id, meta={"store_type": "images",
                                        "results": len(results),
                                        "latency":latency})

        if results:
            log_vision_event(chat_id, "hit", 
                            {"count": len(results), "latency": latency})
        
        else:
            log_vision_event(chat_id, "miss", 
                            {"latency": latency})
        
        return results
    
    except Exception as e:
        log_exception(e, node="retrieve_relevant_images", chat_id=chat_id)
        raise
    

def retrieving_images(query: str):
    q = query.lower()
    return any(key in q for key in Image_Trigger)