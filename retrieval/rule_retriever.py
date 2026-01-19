def retrieve_rules(store, query: str, 
                   competition: str, year: int, 
                   k: int=5):
    return store.similarity_search(query, k=k,
                                   filter={"competition": competition,
                                           "year": year}) # Rule must be filtered before being trusted
