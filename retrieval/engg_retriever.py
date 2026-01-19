def retrieve_engg(store, query: str, k: int=5):
    return store.similarity_search(query, k=k)
