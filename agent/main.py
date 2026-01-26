from typing import Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph
from citations import analyze_rule_docs
from models import AnalyzedRule, CitationModel, AnswerPayload
from agent.subsystem import infer_subsystem
from retrieval.image_retriever import retrieve_relevant_images, retrieving_images

class ChatState(BaseModel):
    question: str
    subqueries: List[str]
    competition: Optional[str]
    year: Optional[int]
    retrieved_rules: List[str]
    retrieved_engg: List[str]
    intent: Optional[str]
    answer: Optional[dict] = None
    design_claims: List[str] = []
    audit_results: Optional[List[dict]] = None
    inspection_history: List[dict] = []
    inspection_stage: int = 0
    inspection_status: str = None
    inspection_strictness: int = 0
    inspection_focus: str = None
    last_user_answer: Optional[str] = None

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

def evaluate_inspection_response(answer: str, strictness: int):
    """ Returns: 'pass', 'ambiguous', 'fail' """
    a = answer.lower()

    if any(key in a for key in ["not sure", "maybe", "depends"]):
        return "fail" if strictness >= 3 else "ambiguous"
    
    if any(key in a for key in ["violates", "not allowed", "non compliant"]):
        return "fail"
    
    if any(key in a for key in ["meets", "complies", "as per the rule"]):
        return "pass"
    
    return "ambiguous"

def inspector_node(state, llm):
    prompt = f"""
You are a Formula Student technical inspector.

Inspection stage: {state.inspection_stage}
Strictness level: {state.inspection_strictness}
Focus area: {state.inspection_focus}

Previous interactions:
{state.inspection_history}

Instructions:
- Ask one precise and most critical, rule-backed inspection question
based on the current discussion.
- Increase strictness if ambiguity was observed.
- Do not explain unless asked.
"""
    
    question = llm.invoke(prompt)
    state.inspection_history.append({"stage": state.inspection_stage,
                                     "question": question})
    state.inspection_stage += 1

    state.answer = {"answer": question,
                    "assumptions": [],
                    "citations": []}
    return state

def inspection_evaluation_node(state: ChatState):
    last_user_answer = state.last_user_answer

    verdict = evaluate_inspection_response(last_user_answer,
                                           state.inspection_strictness)
    if verdict == "fail":
        state.inspection_status = "FAIL"
        state.answer = {"answer": "Inspection failed due to rule non-compliance.",
                        "assumptions": [],
                        "citations": []}
        return state

    if verdict == "pass":
        if state.inspection_strictness >= 3:
            state.inspection_status = "PASS"
            state.answer = {"answer": "Inspection passed. No blocking issues identified.",
                        "assumptions": [],
                        "citations": []}
            return state
        state.inspection_strictness += 1

    state.inspection_strictness += 1
    return state

def compress_rules(rules: list[AnalyzedRule], max_rules=4):
    priority = {"must": 0, "should": 1, "may": 2, "unspecified": 3}
    return sorted(rules, key=lambda r: priority[r.confidence])[:max_rules]

def synthesize_answer(state: ChatState, llm, tools, 
                      clip_model, device):
        
    assumptions = infer_assumptions(state)
    rules = compress_rules(state.retrieved_rules)

    images = []
    if retrieving_images(state.question):
        images = retrieve_relevant_images(query=state["question"],
                                        tools=tools, clip_model=clip_model,
                                        device=device, k=1)

    prompt = f"""

{system_prompt}    

Question:
{state.question}

Rules:
{rules}

Engineering:
{state.retrieved_engineering}

Available diagrams:
{images}

Instructions:
- Clearly distinguish rule requirements from engineering advice
- Do not infer legality beyond provided rules
- Explicitly state assumptions
- Use diagrams only as supporting reference
- Do not invent diagram details

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
        images=images
    )

    state.answer = payload
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

def extract_claims_node(state: ChatState):
    state.design_claims = extract_design_claims(state["question"])
    return state

def audit_node(state: ChatState):
    results = []

    for claim in state.design_claims:
        matched = audit_claim(claim, state.retrieved_rules)
        results.append({"claim":claim,
                        "matched_rules": matched})
    state.audit_results = results
    return state

def synthesize_audit_node(state: ChatState, llm):
    report = synthesize_audit(claims=state.design_claims,
                              matched_rules=state.audit_results,
                              llm=llm)
    state.answer = {"answer": report, 
                    "assumptions": infer_assumptions(state),
                    "citations": []}
    return state 

def build_gragh(llm, tools):
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify_intent)
    graph.add_node("get_rules", lambda s: retrieve_engg_node(s, tools))
    graph.add_node("get_engg", lambda s: retrieve_rule_node(s, tools))
    graph.add_node("answer", lambda s: synthesize_answer(s, llm))

    graph.add_node("extract_claims", extract_claims_node)
    graph.add_node("audit", audit_node)
    graph.add_node("audit_synthesis", lambda s: synthesize_audit_node(s, llm))

    graph.add_node("compare", lambda s: compare_rules_node(s, tools))

    graph.add_node("inspect", lambda s: inspector_node(s, llm))

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", lambda s: s["intent"],
                                {"rule": "get_rules",
                                 "engineering": "get_engg",
                                 "hybrid": "get_rules",
                                 "design_audit": "extract_claims",
                                 "comparison": "compare_rulebooks",
                                 "tech_inspection": "inspect"})
    
    # Standard Q&A edges
    graph.add_edge("get_rules", "get_engg")
    graph.add_edge("get_engg", "answer")

    # Design audit edges
    graph.add_edge("extract_claims", "get_rules")
    graph.add_edge("get_rules", "audit")
    graph.add_edge("audit", "audit_synthesis")

    # Comparison edge
    graph.add_edge("compare", "answer")

    graph.set_finish_point("answer")
    graph.set_finish_point("audit_synthesis")

    return graph.compile()
    