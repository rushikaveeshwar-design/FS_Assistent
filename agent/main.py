import re
from typing import Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph
from citations import analyze_rule_docs
from models import AnalyzedRule, CitationModel, AnswerPayload
from agent.subsystem import infer_subsystem_from_context
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

def decompose_query(query: str, llm) -> list[str]:
    """
    Decompose a complex query into structured subquestions.
    """

    prompt = f"""
You are decomposing a Formula Student engineering query.

Original query:
{query}

Task:
- Identify distinct subquestions.
- Preserve dependencies.
- Rewrite each as a standalone question.
- Keep technical intent intact.

Return output as a JSON list of questions.

"""
    
    result = llm.invoke(prompt)
    return result

def decompose_query_node(state: ChatState, llm):
    state.subqueries = decompose_query(state.question, llm)
    return state

def infer_assumptions(state) -> list[str]:
    assumptions = []

    if not state.year:
        assumptions.append("Latest available rulebook year assumed")

    if not state.competition:
        assumptions.append("Competition inferred from prior context")

    return assumptions

def retrieve_rule_node(state: ChatState, tools, llm):
    """Retrieve and analyze rules using subqueries + subsystem inference."""
    queries = state.subqueries if state.subqueries else [state.question]

    all_docs = []
    seen = set()

    # Infer subsystem using full context
    domain = infer_subsystem_from_context(state.question,
                                          queries, llm)
    for query in queries:
        docs = tools.get_rules(query=query, competition=state.competition,
                               year=state.year, domain=domain)
        
        for doc in docs:
            key = (doc.metadata.get("section"), doc.page_content)
            if key not in seen:
                seen.sdd(key)
                all_docs.append(doc)
    
    # Preserve existing analysis step
    state.retrieve_rules = analyze_rule_docs(all_docs)
    return state

def retrieve_engg_node(state: ChatState, tools):
    docs = tools.get_engg(state["question"])
    state["retrieved_engg"] = docs
    return state

def compare_rules_node(state, tools, llm):
    competitions = state.competitions
    queries = state.subqueries if state.subqueries else [state.question]

    domain = infer_subsystem_from_context(state.question,
                                          queries,llm)

    all_results = {}

    for comp in competitions:
        comp_docs = []

        for query in queries:
            docs = tools.get_rules(
                query=query,
                competition=comp,
                year=state.year,
                domain=domain
            )
            comp_docs.extend(docs)

        all_results[comp] = analyze_rule_docs(comp_docs)

    state.comparison = all_results
    return state

def infer_inspection_focus(*, question: str, subqueries: list[str], 
                           inspection_history: list[dict], llm):
    """
    Infer the most relevant inspection focus (subsystem)
    using semantic reasoning over the inspection context.
    """

    prompt = f"""
You are a fomula Student technical inspector, 
asking relevant and deep questions about engineering and rule compliance.

context:
Main question:
{question}

Decomposed subquestions:
{subqueries}

Inspection history so far:
{inspection_history or "None"}

Task:
Determine the primary subsystem that inspection should focus on NEXT.

Choose ONE from:
- powertrain
- drivetrain (chain sheild, gearbox, differencial system like drive shafts, tripod joints and it's circlips)
- chassis
- safety
- braking
- streeing
- suspension
- aerodynamics
- electronics and battery auxiliaries
- controls and Battery Management System
- general(for eg: Go in detail about the type of nut and bolts used and there industry grade dimensions, etc.)

Rules:
- Base your decision on technical relevance, not keywords.
- If multiple subsystems apply, choose the most safety critical one and then later can move forward with others if necessary.
- If no clear focus exists, return "general(for eg: Go in detail about the type of nut and bolts used and there industry grade dimensions, etc.)"

Return ONLY the subsystem name.

"""
    try:
        focus = llm.invoke(prompt).strip().lower()
        return focus
    except Exception:
        return "general(for eg: Go in detail about the type of nut and bolts used and there industry grade dimensions, etc.)"
    

def evaluate_inspection_response(answer: str, strictness: int, 
                                 referenced_rules: list, llm):
    """ Returns: 'pass', 'ambiguous', 'fail' """

    prompt = f"""You are a Formula Student Technical Inspector.
    
    User response:
    {answer}

    Referenced rules:
    {referenced_rules}

    Strictness level: {strictness}

    Decide:
    - PASS: answer clearly satisfies all rule requirements.
    - AMBIGUOUS: missing values, unclear justification, assumptions.
    - FAIL: contradicts rules or gives invalid specifications.

    Return ONLY one word: PASS, AMBIGUOUS, FAIL.
    """
    verdict = llm.invoke(prompt).strip().upper()
    return verdict.lower()

def extract_rule_refs(text: str) -> list[str]:
    """
    Extract raw rule references from text.
    """
    pattern = r"\b([A-Z]{1,4})[\.\-](\d+(\.\d+)*)\b"
    matches = re.findall(pattern, text)
    return [f"{m[0]}.{m[1]}" for m in matches]

def normalize_rule_refs(raw_refs: list[str], competition: str, 
                        year: int, llm):
    """Validate and normalize rule references."""

    prompt = f"""
You are validating Formula Student rule references.

competition: {competition}
year: {year}

Extracted references:
{raw_refs}

Task:
- Remove invalid references.
- Normalize formatting.
- If ambiguous, keep it but mark as uncertain.

Return as a JSON list of rule references.

"""
    normalized = llm.invoke(prompt)
    return normalized

def enrich_rule_metadata(rule_refs, rule_store, k=1):
    """
    Attach rule metadata from vectorstore.
    """
    enriched = []
    for ref in rule_refs:
        docs = rule_store.similarity_search(ref, k=k)
        if docs:
            enriched.append({"ref": ref,
                             "metadata": docs[0].metadata,
                             "text":docs[0].page_content})
    return enriched

def inspection_focus_node(state, llm):
    state.inspection_focus = infer_inspection_focus(
        question=state.question,
        subqueries=state.subqueries,
        inspection_history=state.inspection_history,
        llm=llm,
    )
    return state

def inspector_node(state, llm, tools, clip_model, device):
    images = retrieve_relevant_images(query=state.inspection_focus or state.question,
                                      tools=tools, clip_model=clip_model,
                                      device=device, k=1)
    prompt = f"""
You are a Formula Student technical inspector.

Inspection stage: {state.inspection_stage}
Strictness level: {state.inspection_strictness}
Focus area: {state.inspection_focus}

Relevant rules:
{state.retrieved_rules}

Relevant diagrams:
{images}

Previous interactions:
{state.inspection_history}

Instructions:
- Ask one precise and most critical, rule-backed inspection question
based on the current discussion.
- The question MUST reference at least one rule section.
- If the diagram is provided frame the question around it.
- Ask the user to justify their design with respect to the diagram.
- Increase strictness if ambiguity was observed.
- Do not explain unless asked.
"""
    
    question = llm.invoke(prompt)
    raw = extract_rule_refs(question)
    normalized = normalize_rule_refs(raw, state.competition, 
                                     state.year, llm)
    rule_refs = enrich_rule_metadata(normalized, tools.rule_store)

    state.inspection_history.append({"stage": state.inspection_stage,
                                     "question": question,
                                     "rule_refs": rule_refs})
    state.inspection_stage += 1

    state.answer = {"answer": question,
                    "assumptions": [],
                    "citations": [],
                    "images": images}
    return state

def inspection_evaluation_node(state: ChatState, llm):
    last_user_answer = state.last_user_answer

    verdict = evaluate_inspection_response(last_user_answer,
                                           state.inspection_strictness,
                                           state.inspection_history[-1].get("rule_refs", []),
                                           llm)
    if verdict == "fail":
        state.inspection_status = "FAIL"
        state.answer = {"answer": "Inspection failed due to rule non-compliance.",
                        "inspection_status": "FAIL",
                        "assumptions": [],
                        "citations": []}
        return state

    if verdict == "pass":
        if state.inspection_strictness >= 3:
            state.inspection_status = "PASS"
            state.answer = {"answer": "Inspection passed. No blocking issues identified.",
                            "inspection_status": "PASS",
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

For EACH design claim, return a structured assessment.

Design claims:
{claims}

Relevant rules:
{matched_rules}

Return output strictly in this format:
[
{{"claim": <original claim>,
    "status": "COMPLIANT | AMBIGUOUS | LIKELY NON-COMPLIANT",
    "reason": "<strong but brief technical reason>",
    "citations": [
    {{"section": "<rule section>",
    "confidence": "<must | should | may>"
    }}
    ]
}}
]


Instructions:
- Do not assume compliance
- Mark each claim as:
COMPLIANT / AMBIGUOUS / LIKELY NON-COMPLIANT
- Cite rule sections
- If ambiguous, state what information is missing
- Do not add extra text outside the structure
"""
    return llm.invoke(prompt)

def extract_design_claims(description: str) -> list[str]:
    lines = description.split("\n")
    return [line.strip() for line in lines if len(line.strip()) > 10]

def audit_claim(claim: str, rule_store, 
                llm, final_k=2, k=3):
    """Retrieve and rerank rules relevant to a design claim."""
    candidate_rules = rule_store.similarity_search(claim, k=k)
    # LLM reranking
    prompt = f"""
You are evaluating rule applicability for a Formula Student Design Audit.

Design claim:
{claim}

Candidate rules:
{[rule.page_content for rule in candidate_rules]}

Task:
Select the {final_k} most relevant rules for judging compliance.
Return ONLY the selected rule texts, in the order of relevance.

"""
    ranked = llm.invoke(prompt)
    selected = []
    for doc in candidate_rules:
        if doc.page_content in ranked:
            selected.append(doc)
    return selected[:final_k]

def extract_claims_node(state: ChatState):
    state.design_claims = extract_design_claims(state["question"])
    return state

def audit_node(state: ChatState, tools, llm):
    results = []

    for claim in state.design_claims:
        matched = audit_claim(claim, tools.rule_store, llm)
        results.append({"claim":claim,
                        "matched_rules": matched})
    state.audit_results = results
    return state

def synthesize_audit_node(state: ChatState, llm):
    structured_audit = synthesize_audit(claims=state.design_claims,
                              matched_rules=state.audit_results,
                              llm=llm)
    state.answer = {"answer": structured_audit, 
                    "assumptions": infer_assumptions(state),
                    "citations": []}
    return state 

def build_gragh(llm, tools):
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify_intent)
    graph.add_node("decompose", lambda s: decompose_query_node(s, llm))
    graph.add_node("get_rules", lambda s: retrieve_engg_node(s, tools))
    graph.add_node("get_engg", lambda s: retrieve_rule_node(s, tools, llm))
    graph.add_node("answer", lambda s: synthesize_answer(s, llm))

    graph.add_node("extract_claims", extract_claims_node)
    graph.add_node("audit", audit_node)
    graph.add_node("audit_synthesis", lambda s: synthesize_audit_node(s, llm))

    graph.add_node("compare", lambda s: compare_rules_node(s, tools, llm))

    graph.add_node("infer_inspection_focus", lambda s: inspection_focus_node(s, llm))
    graph.add_node("inspect", lambda s: inspector_node(s, llm))
    graph.add_node("inspection_evaluation", lambda s: inspection_evaluation_node(s, llm))

    graph.set_entry_point("classify")
    graph.add_edge("classify", "decompose")

    graph.add_conditional_edges("decompose", lambda s: s["intent"],
                                {"rule": "get_rules",
                                 "engineering": "get_engg",
                                 "hybrid": "get_rules",
                                 "design_audit": "extract_claims",
                                 "comparison": "compare_rulebooks",
                                 "tech_inspection": "infer_inspection_focus"})
    
    # Standard Q&A edges
    graph.add_edge("get_rules", "get_engg")
    graph.add_edge("get_engg", "answer")

    # Design audit edges
    graph.add_edge("extract_claims", "get_rules")
    graph.add_edge("get_rules", "audit")
    graph.add_edge("audit", "audit_synthesis")

    # Comparison edge
    graph.add_edge("compare", "answer")

    # Inspection edge
    graph.add_edge("infer_inspection_focus", "inspect")
    graph.add_edge("inspect", "inspection_evaluation")


    graph.set_finish_point("answer")
    graph.set_finish_point("audit_synthesis")
    graph.set_finish_point("inspection_evaluation")

    return graph.compile()
