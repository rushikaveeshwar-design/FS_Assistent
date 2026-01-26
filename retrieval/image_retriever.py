from data_ingestion.clip_embedder import embed_text_for_clip

Image_Trigger = ["diagram", "figure", "layout", "image"]

def retrieve_relevant_images(query: str, tools, clip_model,
                             device, k: int=1):
    query_vector = embed_text_for_clip(query, model=clip_model,
                                       device=device)
    
    docs = tools.get_images_by_vector(query_vector, k=k)

    results = []
    for d in docs:
        metadata = d.metadata
        results.append({"source": metadata["source"],
                        "page": metadata["page"],
                        "domain": metadata["domain"],
                        "competition": metadata["competition"],
                        "year": metadata["year"],
                        "section": metadata["section"],
                        "caption": metadata["caption"]})
    return results

def retrieving_images(query: str):
    q = query.lower()
    return any(key in q for key in Image_Trigger)