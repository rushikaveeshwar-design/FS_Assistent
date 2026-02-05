from retrieval.rule_retriever import retrieve_rules
from retrieval.engg_retriever import retrieve_engg
from typing import Generator, Dict
from agent.main import build_gragh

class AgentTool:
    def __init__(self, rule_store, engg_store, image_store,
                 vectorstore_manager, chat_store):
        self.rule_store = rule_store
        self.engg_store = engg_store
        self.image_store = image_store
        self.vectorstore_manager = vectorstore_manager
        self.chat_store = chat_store

    def get_rules(self, query, competition, year, domain=None):
        return retrieve_rules(store=self.rule_store,
                              query=query, competition=competition,
                              year=year, domain=domain)
    
    def get_engg(self, query):
        return retrieve_engg(store=self.engg_store, query=query)
    
    def get_image_by_vector(self, query_vector, k=1):
        if not self.image_store:
            return []
        return self.image_store.similarity_search_by_vector(query_vector, k=k)

def compile_graph(llm, embedding_model, clip_embedding_model,
                  vectorstore_manager, chat_store):

    tools = AgentTool(embedding_model=embedding_model, clip_embedding_model=clip_embedding_model, 
                      vectorstore_manager=vectorstore_manager, 
                      chat_store=chat_store)
    return build_gragh(llm, tools)
    

def run_agent_stream(*, question: str, mode: str,
                     competition: str | None, year: int | None,
                     chat_id: str, llm, embedding_model, clip_embedding_model,
                     vectorstore_manager, chat_store) -> Generator[Dict, None, None]:
    """Yields incremental agent output.
    
    Final yield MUST contain:
    {"answer": str, "assumptions": list[str],
    "citations": list[dict], "images": list[dict] | None}

    """

    compiled_graph = compile_graph(llm, embedding_model=embedding_model,
                                   clip_embedding_model=clip_embedding_model,
                                   vectorstore_manager=vectorstore_manager, chat_store=chat_store)

    # Initial state
    state = {"question": question, "mode": mode,
             "competition": competition, "year": year,
             "chat_id": chat_id}

    # streaming
    for update in compiled_graph.stream(state):
        # Only have to stream answer tokens
        if "answer_token" in update:
            yield {"token": update["answer_token"]}

        if "final_answer" in update:
            final_payload = update["final_answer"]

            if "images" not in final_payload:
                final_payload["images"] = []

            yield final_payload
