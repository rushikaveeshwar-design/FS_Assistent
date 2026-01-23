from retrieval.rule_retriever import retrieve_rules
from retrieval.engg_retriever import retrieve_engg
from typing import Generator, Dict
from agent.main import build_gragh
from retrieval.vectorstore import load_rule_store, load_engg_store

class AgentTool:
    def __init__(self, rule_store, engg_store):
        self.rule_store = rule_store
        self.engg_store = engg_store

    def get_rules(self, query, competition, year, domain=None):
        return retrieve_rules(store=self.rule_store,
                              query=query, competition=competition,
                              year=year, domain=domain)
    
    def get_engg(self, query):
        return retrieve_engg(store=self.engg_store, query=query)

def compile_graph(llm):
    rule_store = load_rule_store()
    engg_store = load_engg_store()

    tools = AgentTool(rule_store, engg_store)
    return build_gragh(llm, tools)
    

def run_agent_stream(*, question: str, mode: str,
                     competition: str | None, year: int | None,
                     chat_id: str, llm) -> Generator[Dict, None, None]:
    """Yields incremental agent output.
    
    Final yield MUST contain:
    {"answer": str, "assumption": list[str],
    "citations": list[dict]}

    """

    compiled_graph = compile_graph(llm)

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
            yield update["final_answer"]
