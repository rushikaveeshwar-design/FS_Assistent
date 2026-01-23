from typing import Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph
from citations import analyze_rule_docs
from models import AnalyzedRule, CitationModel, AnswerPayload
from agent.subsystem import infer_subsystem

class ChatState(BaseModel):
    question: str
    subqueries: List[str]
    competition: Optional[str]
    year: Optional[int]
    retrieved_rules: List[str]
    retrieved_engg: List[str]
    intent: Optional[str]
    answer: Optional[dict] = None

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
    elif "review" in q or "check" in q or "audit" in q:
        state["intent"] = "design_audit"
    elif "inspection" in q or "scrutineering" in q:
        state["intent"] = "tech_inspection"
    else:
        state["intent"] = "hybrid"
    return state

def infer_assumptions(state) -> list[str]:
    assumptions = []

    if not state.year:
        assumptions.append("Latest available rulebook year assumed")

    if not state.competition:
        assumptions.append("Competition inferred from prior context")

    return assumptions

def retrieve_rule_node(state: ChatState, tools):
    domain = infer_subsystem(state["question"])
    docs = tools.get_rules(query=state["question"],
                           competition=state["competition"],
                           year=state["year"],
                           domain=domain)
    state["retrieve_rules"] = analyze_rule_docs(docs)
    return state

def retrieve_engg_node(state: ChatState, tools):
    docs = tools.get_engg(state["question"])
    state["retrieved_engg"] = docs
    return state

def compare_rules_node(state, tools):
    competitions = state["competitions"] # that's a list
    all_results = {}
    domain = infer_subsystem(state["question"])

    for comp in competitions:
        docs = tools.get_rules(query=state["question"],
                               competition=comp,
                               year=state["year"],
                               domain=domain)
        all_results[comp] = analyze_rule_docs(docs)
    
    state["comparison"] = all_results
    return state

def inspector_node(state, llm):
    prompt = """
You are a Formula Student technical inspector.

Ask one precise, rule-backed question
based on the current discussion.
Do not explain unless asked.
"""
    
    question = llm.invoke(prompt)
    state["inspector_question"] = question
    return state

def compress_rules(rules: list[AnalyzedRule], max_rules=4):
    priority = {"must": 0, "should": 1, "may": 2, "unspecified": 3}
    return sorted(rules, key=lambda r: priority[r.confidence])[:max_rules]

def synthesize_answer(state: ChatState, llm):
        
    assumptions = infer_assumptions(state)
    rules = compress_rules(state.retrieved_rules)

    prompt = f"""

{system_prompt}    

Question:
{state.question}

Rules:
{rules}

Engineering:
{state.retrieved_engineering}

Instructions:
- Clearly distinguish rule requirements from engineering advice
- Do not infer legality beyond provided rules
- explicitly state assumptions

Provide a clear, grounded answer:
"""
    answer_text = ""

    for token in llm.stream(prompt):
        answer_text += token
        yield {"answer_token": token}


    citations = [
        CitationModel(
            competition=r.metadata.competition,
            year=r.metadata.year,
            section=r.metadata.section,
            source=r.metadata.source,
            confidence=r.confidence,
        ) # Do we want here a dict?!
        for r in rules
    ]

    payload = AnswerPayload(
        answer=answer_text,
        citations=citations,
        assumptions=assumptions,
    )

    state.answer = payload.dict()
    return state, {"final_answer": payload}

def synthesize_audit(claims, matched_rules, llm):
    prompt = f"""
You are reviewing a Formula Student design for rule compliance.

Design claims:
{claims}

Relevant rules:
{matched_rules}

Instructions:
- Do not assume compliance
- Mark each claim as:
COMPLIANT / AMBIGUOUS / LIKELY NON-COMPLIANT
- Cite rule sections
- If ambiguous, state what information is missing
"""
    return llm.invoke(prompt)

def decompose_query(query: str) -> list[str]:
    separators = [" and ", ",", ";"]
    subqueries = [query]

    for separator in separators:
        new = []
        for q in subqueries:
            new.extend(q.split(separator))
        subqueries = new
    return [q.strip() for q in subqueries if q.strip()]

def extract_design_claims(description: str) -> list[str]:
    lines = description.split("\n")
    return [line.strip() for line in lines if len(line.strip()) > 10]

def audit_claim(claim: str, rules):
    findings = []

    for rule in rules:
        if any(word in rule.lower() for word in claim.lower().split()):
            findings.append(rule)
    return findings

def build_gragh(llm, tools):
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify_intent)
    graph.add_node("get_rules", lambda s: retrieve_engg_node(s, tools))
    graph.add_node("get_engineering", lambda s: retrieve_rule_node(s, tools))
    graph.add_node("answer", lambda s: synthesize_answer(s, llm))
    graph.add_node("inspect", lambda s: inspector_node(s, llm))

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", lambda s: s["intent"],
                                {"rule": "get_rules",
                                 "engineering": "get_engg",
                                 "hybrid": "get_rules",
                                 "design_audit": "audit",
                                 "compare": "compare_rulebook",
                                 "tech_inspection": "inspect"})
    
    graph.add_edge("get_rules", "get_engg")
    graph.add_edge("get_engg", "answer")

    graph.set_finish_point("answer")
    return graph.compile()
    