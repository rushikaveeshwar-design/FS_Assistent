from typing import TypedDict, List
from langgraph.graph import StateGraph
from citations import analyze_rule_docs

class ChatState(TypedDict):
    question: str
    competition: str
    year: int
    retrieved_rules: List[str]
    retrieved_engg: List[str]
    answer: str

system_prompt = """
You are a formula student engineering assistant.

Rules:
- Never merge rulebooks from different competitions unless explicitly asked.
- Always prefer official rules over engineering best practices when compliance is involved, but you can just state that engineering practice.
- If unsure, state assumptions clearly.
- Cite rule sections when available.
- Engineering explanations must be practical and student-oriented.

You are allowed to:
-Compare competitions when required.
- Explain engineering tradeoffs.
- highlight conflicts between rules and best practices
- If there is any need of formula explanation and concept clarification, the explanation should be in depth and very clear.

"""

def classify_intent(state: ChatState):
    q = state["question"].lower()

    if "rule" in q or "allowed" in q or "legal" in q:
        state["intent"] = "rule"
    elif "design" in q or "how to" in q or "explain" in q:
        state["intent"] = "engineering"
    elif "compare" in q or "difference" in q:
        state["intent"] = "comparison"
    else:
        state["intent"] = "hybrid"
    return state

def retrieve_rule_node(state: ChatState, tools):
    docs = tools.get_rules(query=state["question"],
                           competition=state["competition"],
                           year=state["year"])
    state["retrieve_rules"] = analyze_rule_docs(docs)
    return state

def retrieve_engg_node(state: ChatState, tools):
    docs = tools.get_engg(state["question"])
    state["retrieved_engg"] = docs
    return state

def compare_rules_node(state, tools):
    competitions = state["competitions"] # that's a list
    all_results = {}

    for comp in competitions:
        docs = tools.get_rules(state["question"],
                               competition=comp,
                               year=state["year"])
        all_results[comp] = analyze_rule_docs(docs)
    
    state["comparison"] = all_results
    return state

def synthesize_answer(state: ChatState, llm):
    rule_context = ""
    citations = []

    for r in state.get("retrieved_rules",[]):
        rule_context += f"\n[{r['metadata']['section']}]\n{r['text']}\n"
        citations.append({"source": r['metadata']['source'],
                          "competition": r['metadata']['competition'],
                          "year": r['metadata']['year'],
                          "section": r['metadata']['section'],
                          "confidence": r['confidence']})
        
    engg_context = ""
    for e in state.get("retrieved_engg", []):
        engg_context += f"\n{e.page_content}\n"
    
    prompt = f"""
Question:
{state["question"]}

RULES (authoritative):
{rule_context}

ENGINEERING REFERENCES (non-authoritative):
{engg_context}

Instructions:
- Clearly distinguish rule requirements from engineering advice
- Do not infer legality beyond provided rules
- explicitly state assumptions

Provide a clear, grounded answer:
"""
    answer = llm.invoke(prompt)

    state["answer"] = {"text": answer,
                       "citations": citations}
    return state

def build_gragh(llm, tools):
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify_intent)
    graph.add_node("get_rules", lambda s: retrieve_engg_node(s, tools))
    graph.add_node("get_engineering", lambda s: retrieve_rule_node(s, tools))
    graph.add_node("answer", lambda s: synthesize_answer(s, llm))

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", lambda s: s["intent"],
                                {"rule": "get_rules",
                                 "engineering": "get_engg",
                                 "hybrid": "get_rules"})
    
    graph.add_edge("get_rules", "get_engg")
    graph.add_edge("get_engg", "answer")

    graph.set_finish_point("answer")
    return graph.compile()
    