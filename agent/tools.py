from retrieval.rule_retriever import retrieve_rules
from retrieval.engg_retriever import retrieve_engg

class AgentTool:
    def __init__(self, rule_store, engg_store):
        self.rule_store = rule_store
        self.engg_store = engg_store

    def get_rules(self, query, competition, year):
        return retrieve_rules(store=self.rule_store,
                              query=query, competition=competition,
                              year=year)
    
    def get_engg(self, query):
        return retrieve_engg(store=self.engg_store, query=query)
    
    