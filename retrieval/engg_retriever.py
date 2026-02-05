def retrieve_engg(store, query: str, k: int=3):
    return store.similarity_search(query, k=k)
