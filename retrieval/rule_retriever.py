def retrieve_rules(store, query: str, 
                   competition: str, year: int, 
                   domain: str | None=None, k: int=5):
    filters = {"competition": competition,
               "year": year}
    if domain:
        filters["domain"] = domain

    return store.similarity_search(query, k=k,
                                   filter=filters) # Rule must be filtered before being trusted
