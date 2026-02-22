import re
from datetime import datetime, timezone
import time
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from agent.citations import analyze_rule_docs
from agent.models import AnalyzedRule, CitationModel, AnswerPayload
from agent.tools import extract_visual_observations, build_diagram_instructions, infer_diagram_relevance
from agent.subsystem import infer_subsystem_from_context
from retrieval.image_retriever import retrieve_relevant_images, retrieving_images
from data_ingestion.doc_utils import (load_document_registry, resolve_rulebook_documents,
                                      resolve_engineering_documents)
from data_ingestion.build_vectorstore import load_rule_vectorstores, load_engg_vectorstores, load_image_vectorstores
from retrieval.rule_retriever import retrieve_rules
from retrieval.engg_retriever import retrieve_engg
from agent.logger import (log_event, log_node_enter, log_node_exit, log_exception,
                          invoke_llm, stream_llm)

class FocusObject(BaseModel):
    focus_id: str
    origin: str
    status: str = "UNRESOLVED"
    question_asked: List[str] = Field(default_factory=list)
    answers_received: List[str] = Field(default_factory=list)
    referenced_rules: List[dict] = Field(default_factory=list)
    referenced_images: List[dict] = Field(default_factory=list)
    evidence_notes: List[str] = Field(default_factory=list)

class InspectionState(BaseModel):
    active: bool = False
    focus_stack: List[FocusObject] = Field(default_factory=list)
    active_focus_index: int = 0
    interaction_history: List[dict] = Field(default_factory=list)
    global_verdict: Optional[str] = None

class ChatState(BaseModel):
    question: str
    mode: str
    subqueries: List[str] = Field(default_factory=list)
    competition: Optional[str]
    year: Optional[int]
    retrieved_rules: List[str] = Field(default_factory=list)
    retrieved_engg: List[str] = Field(default_factory=list)
    intent: Optional[str]
    answer: Optional[dict] = None
    design_claims: List[str] = Field(default_factory=list)
    audit_results: Optional[List[dict]] = None
    inspection: InspectionState = InspectionState()
    user_images: List = Field(default_factory=list)
    image_observations: List[str] = Field(default_factory=list)
    chat_id: Optional[str] = None

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
    q = state.question.lower()

    if "rule" in q or "allowed" in q or "legal" in q:
        state.intent = "rule"
    elif "design" in q or "how to" in q or "explain" in q:
        state.intent = "engineering"
    elif "compare" in q or "difference" in q:
        state.intent = "comparison"
    elif "review" in q or "check" in q or "audit" in q:
        state.intent = "design_audit"
    elif "inspection" in q or "scrutineering" in q:
        state.intent = "tech_inspection"
    else:
        state.intent = "hybrid"
    return state

def decompose_query(state: ChatState, query: str, llm) -> list[str]:
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
    
    result = invoke_llm(llm, prompt, chat_id=state.chat_id, 
                        node="decompose_query")
    return result

def decompose_query_node(state: ChatState, llm):
    state.subqueries = decompose_query(state, state.question, llm)
    return state

def infer_assumptions(state) -> list[str]:
    assumptions = []

    if not state.year:
        assumptions.append("Latest available rulebook year assumed")

    if not state.competition:
        assumptions.append("Competition inferred from prior context")

    return assumptions

def retrieve_rule_node(state: ChatState, tools, llm):
    """Retrieve and analyze rule documents.
    This node:
    - Resolves relevant rulebook PDFs from registry
    - Loads/creates TEXT vectorstores
    - Loads/creates IMAGE vectorstores from the same PDFs
    - Attaches both the chat
    - Retrieves rule chunks using subqueries
    """
    log_node_enter("retrieve_rule_node", state.chat_id)

    try:

        t0 = time.time()
        queries = state.subqueries if state.subqueries else [state.question]

        registry = load_document_registry()
        descriptors = resolve_rulebook_documents(registry, competition=state.competition,
                                                year=state.year)
        
        rule_stores = load_rule_vectorstores(descriptors=descriptors, 
                                            chat_id=state.chat_id, chat_store=tools.chat_store,
                                            vectorstore_manager=tools.vectorstore_manager)
        
        image_stores = load_image_vectorstores(descriptors=descriptors, chat_id=state.chat_id,
                                            chat_store=tools.chat_store,
                                            vectorstore_manager=tools.vectorstore_manager,
                                            clip_model=tools.clip_model, preprocess=tools.preprocess, device=tools.device)
        
        # Attach stores to tools
        tools.rule_store = rule_stores
        tools.image_store = image_stores

        all_docs = []
        seen = set()

        # Infer subsystem using full context
        memory = tools.memory_engine.load(state.chat_id)

        domain = infer_subsystem_from_context(state, question=state.question,
                                              subqueries=queries, llm=llm,
                                              memory=memory)
        
        for store in rule_stores:
            for query in queries:
                docs = retrieve_rules(store=store, query=query, competition=state.competition,
                                    year=state.year, domain=domain)
                
                for doc in docs:
                    key = (doc.metadata.get("section"), doc.page_content)
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
        
        state.retrieved_rules = analyze_rule_docs(all_docs)

        latency = int((time.time() - t0)*1000)

        log_event("INFO", "rule_retrieval_done",
                  chat_id=state.chat_id, meta={"docs": len(all_docs),
                                               "num_stores": len(rule_stores),
                                               "latency": latency})
        
        log_node_exit("retrieve_rule_node", state.chat_id)
        return state
    
    except Exception as e:
        log_exception(e, chat_id=state.chat_id, node="retrieve_rule_node")
        raise

def retrieve_engg_node(state: ChatState, tools, llm):
    """Retrieve and analyze engineering documents.
    This node:
    - Resolves relevant engineering PDFs from registry
    - Loads/creates TEXT vectorstores
    - Loads/creates IMAGE vectorstores from same PDFs
    - Attaches both to the chat
    - Retrieves engineering chunks
    """
    log_node_enter("retrieve_engg_node", state.chat_id)

    try:

        t0 = time.time()

        queries = state.subqueries if state.subqueries else [state.question]

        memory = tools.memory_engine.load(state.chat_id)

        registry = load_document_registry()

        descriptors = resolve_engineering_documents(registry,
                                                    domain=infer_subsystem_from_context(state, question=state.question, 
                                                                                        subqueries=queries, llm=llm, 
                                                                                        memory=memory),
                                                    topic=None)
        
        engg_stores = load_engg_vectorstores(descriptors=descriptors,
                                            chat_id=state.chat_id, chat_store=tools.chat_store,
                                            vectorstore_manager=tools.vectorstore_manager)
        
        image_stores = load_image_vectorstores(descriptors=descriptors, 
                                            chat_id=state.chat_id,
                                            vectorstore_manager=tools.vectorstore_manager,
                                            chat_store=tools.chat_store, clip_model=tools.clip_model,
                                            preprocess=tools.preprocess, device=tools.device)
        
        tools.engg_store = engg_stores
        tools.image_store = image_stores
        
        docs = []
        for store in engg_stores:
            docs.extend(retrieve_engg(store=store, query=state.question))
        state.retrieved_engg = docs

        latency = int((time.time() - t0)*1000)

        log_event("INFO", "engg_retrieval_done",
                chat_id=state.chat_id,
                meta={"docs": len(docs),
                        "stores": len(engg_stores),
                        "latency": latency})
        
        log_node_exit("retrieve_engg_node", state.chat_id)
        return state
    
    except Exception as e:
        log_exception(e, chat_id=state.chat_id, node="retrieve_engg_node")
        raise

# Compare rules
def compare_rules_node(state: ChatState, tools, llm):
    competitions = state.competition
    queries = state.subqueries if state.subqueries else [state.question]

    memory = tools.memory_engine.load(state.chat_id)

    domain = infer_subsystem_from_context(state, question=state.question,
                                          subqueries=queries,llm=llm,
                                          memory=memory)

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

def infer_inspection_focus(*, state: ChatState, question: str, subqueries: list[str], 
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
        focus = invoke_llm(llm, prompt, chat_id=state.chat_id,
                           node="infer_inspection_focus")
        focus = focus.strip().lower()
        return focus
    except Exception:
        return "general(for eg: Go in detail about the type of nut and bolts used and there industry grade dimensions, etc.)"
    
def should_shift_focus(current_focus: str, inferred_focus: str,
                       focus_stack):
    if inferred_focus == current_focus:
        return False
    
    for fs in focus_stack:
        if fs.focus_id == inferred_focus and fs.status == "UNRESOLVED":
            return False
    
    return True

def next_unresolved_focus_index(focus_stack):
    for idx, fs in enumerate(focus_stack):
        if fs.status == "UNRESOLVED":
            return idx
    return None

def compute_global_verdict(focus_stack):
    failed = [fs for fs in focus_stack if fs.status == "FAILED"]
    if not failed:
        return "PASSED"
    
    for f in failed:
        if f.focus_id in {"bracking", "battery", "safety"}:
            return "FAILED"
    
    return "REWORK REQUIRED"

def evaluate_inspection_response(state: ChatState, answer: str, focus_id: str, 
                                 referenced_rules: list, llm):
    """ Returns: 'pass', 'ambiguous', 'fail' """

    prompt = f"""You are a Formula Student Technical Inspector.
    
    focus area: {focus_id}

    User response:
    {answer}

    Referenced rules:
    {referenced_rules}

    Evaluate strictly based on rule compliance and engineering correctness.

    Decision criteria:
    - PASS: answer clearly satisfies all rule requirements.
    - AMBIGUOUS: missing values, unclear justification, assumptions.
    - FAIL: contradicts rules or gives invalid specifications.

    Return ONLY one word: PASS, AMBIGUOUS, FAIL.
    """
    verdict = invoke_llm(llm, prompt, chat_id=state.chat_id,
                         node="evaluate_inspection_response")
    verdict = verdict.strip().upper()
    return verdict.lower()

def extract_rule_refs(text: str) -> list[str]:
    """
    Extract raw rule references from text.
    """
    pattern = r"\b([A-Z]{1,4})[\.\-](\d+(\.\d+)*)\b"
    matches = re.findall(pattern, text)
    return [f"{m[0]}.{m[1]}" for m in matches]

def normalize_rule_refs(state: ChatState, raw_refs: list[str], competition: str, 
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
    normalized = invoke_llm(llm, prompt, chat_id=state.chat_id,
                            node="normalize_rule_refs")
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

def infer_primary_focus_node(state: ChatState, llm):

    if state.inspection.active:
        return state
    
    focus = infer_inspection_focus(question=state.question,
                                   subqueries=state.subqueries,
                                   inspection_history=[],
                                   llm=llm)
    
    state.inspection.active = True
    state.inspection.focus_stack.append(FocusObject(focus_id=focus,
                                                    origin="primary"))
    state.inspection.active_focus_index = 0
    return state

# Inspection mode
def inspector_node(state: ChatState, llm, tools, clip_model, device):
    inspection = state.inspection
    focus = inspection.focus_stack[inspection.active_focus_index]

    images = []
    if retrieving_images(state.question):
        images = retrieve_relevant_images(query=focus.focus_id or state.question,
                                        tools=tools, clip_model=clip_model,
                                        device=device, k=1, chat_id=state.chat_id)
        
    diagram_mode = infer_diagram_relevance(mode="Tech Inspection", question=state.question,
                                           retrieved_rules=state.retrieved_rules, retrieved_engg=state.retrieved_engg,
                                           images=images)
    
    diagram_instructions = build_diagram_instructions(diagram_mode=diagram_mode,
                                                      images=images)
    
    visual_context = state.image_observations or []

    memory = tools.memory_engine.load(state.chat_id)
    
    prompt = f"""
You are a Formula Student technical inspector.

Current focus: {focus.focus_id}

Relevant rules:
{state.retrieved_rules}

Known project context:
{memory}

Relevant diagrams:
{images or "None"}

User submitted visual context:
{visual_context}

Previous interactions:
{inspection.interaction_history}

{diagram_instructions}

Instructions:
- Ask one precise and most critical, rule-backed inspection question
based on the current discussion.
- The question MUST reference at least one rule section.
- If the diagram is provided, frame the question around it.
- Ask the user to justify their design with respect to the diagram, if diagram available.
- Increase strictness if ambiguity was observed.
- Do not explain unless asked.
- Use user visual observations ONLY as supporting evidence
- If user visual interpretation is uncertain, state it
- Do not assume compliance from image alone
"""
    
    question = invoke_llm(llm, prompt, chat_id=state.chat_id,
                          node="inspector_node")
    raw = extract_rule_refs(question)
    normalized_refs = normalize_rule_refs(state, raw, state.competition, 
                                     state.year, llm)
    enriched_rule_refs = enrich_rule_metadata(normalized_refs, tools.rule_store)

    # Persist evidence into focus
    focus.question_asked.append(question)
    focus.referenced_rules.extend(enriched_rule_refs)
    focus.referenced_images.extend(images)

    inspection.interaction_history.append({
        "focus": focus.focus_id,
        "question": question,
        "rules": normalized_refs
    })

    state.answer = {"answer": question,
                    "assumptions": [],
                    "citations": [],
                    "images": images}
    
    return state

def inspection_evaluation_node(state: ChatState, llm):

    inspection = state.inspection
    focus = inspection.focus_stack[inspection.active_focus_index]

    user_answer = state.question # user response as state.question
    focus.answers_received.append(user_answer)

    verdict = evaluate_inspection_response(answer=user_answer, focus_id=focus.focus_id,
                                           referenced_rules=focus.referenced_rules,
                                           llm=llm)
    
    log_event("INFO", "inspection_focus_verdict", chat_id=state.chat_id, 
              meta={"focus": focus.focus_id, 
                    "verdict": verdict})

    inspection.interaction_history.append({"focus": focus.focus_id,
                                           "question": focus.question_asked[-1] if focus.question_asked else None,
                                           "answer": user_answer,
                                           "verdict": verdict})

    if verdict == "pass":
        focus.status = "PASSED"
    elif verdict == "fail":
        focus.status = "FAILED"

    # next transition decision block
    if verdict == "ambiguous":
        return state
    
    next_index = next_unresolved_focus_index(inspection.focus_stack)

    if not None and next_index < len(inspection.focus_stack):
        inspection.active_focus_index = next_index
        return state
    
    # Compute global verdict
    inspection.global_verdict = compute_global_verdict(inspection.focus_stack)

    inspection.active = False

    report = build_inspection_report(inspection)

    log_event("INFO", "inspection_complete", chat_id=state.chat_id,
              meta={"global_verdict": inspection.global_verdict})

    state.answer = {"answer": f"Inspection complete. Final verdict: {inspection.global_verdict}",
                    "inspection_complete": True,
                    "global_verdict": inspection.global_verdict,
                    "report": report,
                    "per_focus_results": [{"focus": fs.focus_id,
                                           "status": fs.status}
                                           for fs in inspection.focus_stack],
                    "assumptions":[],
                    "citations":[]}
    
    return state

def inspection_focus_router_node(state: ChatState, llm):
    inspection = state.inspection

    inferred_focus = infer_inspection_focus(question=state.question, subqueries=state.subqueries,
                                      inspection_history=inspection.interaction_history,
                                      llm=llm)
    
    current_focus = inspection.focus_stack[inspection.active_focus_index].focus_id

    if should_shift_focus(current_focus, inferred_focus,
                          inspection.focus_stack):
        inspection.focus_stack.append(FocusObject(focus_id=inferred_focus,
                                                  origin="secondary"))
        
        inspection.active_focus_index = len(inspection.focus_stack) - 1
        return state
    
    next_idx = next_unresolved_focus_index(inspection.focus_stack)

    if next_idx is not None:
        inspection.active_focus_index = next_idx
        return state
    
    inspection.active = False
    return state

# Very naive approach
def generate_confidence_explanation(focus: FocusObject):

    if focus.status == "PASSED":
        return ("All referenced rule requirements are explicitly addressed"
                "with no contradictions detected.")
    
    if focus.status == "FAILED":
        return ("Atleast one referenced rule requirement was violated"
                "or contradicted with the provided explanation.")
    
    return ("Inspection remained inconclusive due to missing data or"
            "insufficient justification.")

def build_inspection_report(inspection: InspectionState):
    per_focus = []

    for focus in inspection.focus_stack:
        per_focus.append({"focus_id": focus.focus_id,
                          "origin": focus.origin,
                          "status": focus.status,
                          "questions": focus.question_asked,
                          "answers": focus.answers_received,
                          "referenced_rules": focus.referenced_rules,
                          "referenced_images": focus.referenced_images,
                          "confidence_explanation": generate_confidence_explanation(focus)})
    
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "global_verdict": inspection.global_verdict,
            "focus_count": len(inspection.focus_stack),
            "per_focus": per_focus,
            "interaction_history": inspection.interaction_history}

def compress_rules(rules: list[AnalyzedRule], max_rules=4):
    priority = {"must": 0, "should": 1, "may": 2, "unspecified": 3}
    return sorted(rules, key=lambda r: priority[r.confidence])[:max_rules]

# Rule Q&A
def synthesize_answer(state: ChatState, llm, tools, 
                      clip_model, device):
        
    assumptions = infer_assumptions(state)
    rules = compress_rules(state.retrieved_rules)

    images = []
    if retrieving_images(state.question):
        images = retrieve_relevant_images(query=state.question,
                                        tools=tools, clip_model=clip_model,
                                        device=device, k=1, chat_id=state.chat_id)
        
    diagram_mode = infer_diagram_relevance(mode=state.mode, question=state.question,
                                           retrieved_engg=state.retrieved_engg,
                                           retrieved_rules=rules, images=images)
    
    diagram_instructions = build_diagram_instructions(diagram_mode=diagram_mode,
                                                      images=images)
    
    visual_context = state.image_observations or []

    memory = tools.memory_engine.load(state.chat_id)

    prompt = f"""

{system_prompt}    

Question:
{state.question}

Rules:
{rules}

Engineering:
{state.retrieved_engg}

Known project context:
{memory}

User uploaded visual context:
{visual_context}

Available diagrams:
{images or "None"}

{diagram_instructions}

Instructions:
- Clearly distinguish rule requirements from engineering advice
- Do not infer legality beyond provided rules
- Explicitly state assumptions
- Use diagrams only as supporting reference
- Do not invent diagram details
- Use user visual observations ONLY as supporting evidence
- If user visual interpretation is uncertain, state it
- Do not assume compliance from image alone

Provide a clear, grounded answer:
"""
    answer_text = ""

    for token in stream_llm(llm, prompt, chat_id=state.chat_id,
                            node="synthesize_answer"):
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

    payload_dict = payload.model_dump()
    state.answer = payload_dict
    return state, {"final_answer": payload_dict}

def synthesize_audit(state: ChatState, claims, matched_rules, llm, tools):
    memory = tools.memory_engine.load(state.chat_id)

    prompt = f"""
You are reviewing a Formula Student design for rule compliance.

For EACH design claim, return a structured assessment.

Design claims:
{claims}

Relevant rules:
{matched_rules}

Known project context:
{memory}

User visual context:
{state.image_observations}

Diagram handling:
- If diagrams are referenced in claims or rules but not provided,
  mark the claim as AMBIGUOUS.
- Do NOT assume geometry or layout.
- Cite rules precisely.

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
- Use user visual observations ONLY as supporting evidence
- If user visual interpretation is uncertain, state it
- Do not assume compliance from image alone
"""
    return invoke_llm(llm, prompt, chat_id=state.chat_id,
                      node="synthesize_audit")

def extract_design_claims(description: str) -> list[str]:
    lines = description.split("\n")
    return [line.strip() for line in lines if len(line.strip()) > 10]

def audit_claim(state: ChatState, claim: str, rule_store, 
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
    ranked = invoke_llm(llm, prompt, chat_id=state.chat_id, 
                        node="audit_claim")
    selected = []
    for doc in candidate_rules:
        if doc.page_content in ranked:
            selected.append(doc)
    return selected[:final_k]

def extract_claims_node(state: ChatState):
    state.design_claims = extract_design_claims(state.question)
    return state

def audit_node(state: ChatState, tools, llm):
    results = []

    for claim in state.design_claims:
        matched = audit_claim(claim, tools.rule_store, llm)
        results.append({"claim":claim,
                        "matched_rules": matched})
    state.audit_results = results
    return state

# Design audit
def synthesize_audit_node(state: ChatState, llm):
    structured_audit = synthesize_audit(claims=state.design_claims,
                              matched_rules=state.audit_results,
                              llm=llm)
    state.answer = {"answer": structured_audit, 
                    "assumptions": infer_assumptions(state),
                    "citations": []}
    return state

def vision_analysis_node(state: ChatState, vision_llm):
    """Extracts visual observations from user images."""

    if not state.user_images:
        state.image_observations = []
        return state
    
    state.image_observations = extract_visual_observations(state.user_images, 
                                                           vision_llm=vision_llm)
    return state

def build_graph(llm, tools, vision_llm, clip_model, device):
    graph = StateGraph(ChatState)

    graph.add_node("classify", lambda s: classify_intent(s))
    graph.add_node("decompose", lambda s: decompose_query_node(s, llm))
    graph.add_node("get_rules", lambda s: retrieve_rule_node(s, tools, llm))
    graph.add_node("get_engg", lambda s: retrieve_engg_node(s, tools, llm))
    graph.add_node("answer", lambda s: synthesize_answer(s, llm, tools, clip_model, device))

    graph.add_node("extract_claims", lambda s: extract_claims_node(s))
    graph.add_node("audit", lambda s: audit_node(s, tools, llm))
    graph.add_node("audit_synthesis", lambda s: synthesize_audit_node(s, llm))

    graph.add_node("compare", lambda s: compare_rules_node(s, tools, llm))

    graph.add_node("infer_inspection_focus", lambda s: infer_primary_focus_node(s, llm))
    graph.add_node("inspect", lambda s: inspector_node(s, llm, tools, clip_model, device))
    graph.add_node("inspection_evaluation", lambda s: inspection_evaluation_node(s, llm))
    graph.add_node("inspection_router", lambda s: inspection_focus_router_node(s, llm))

    graph.add_node("vision_analysis", lambda s: vision_analysis_node(s, vision_llm))

    graph.set_entry_point("classify")
    graph.add_edge("classify", "decompose")

    graph.add_edge("decompose", "vision_analysis")

    graph.add_conditional_edges("vision_analysis", lambda s: s["intent"],
                                {"rule": "get_rules",
                                 "engineering": "get_engg",
                                 "hybrid": "get_rules",
                                 "design_audit": "extract_claims",
                                 "comparison": "compare",
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
    graph.add_edge("inspection_evaluation", "inspection_router")

    graph.add_conditional_edges("inspection_router",
                                lambda s: s.inspection.active,
                                {True: "inspect",
                                 False: "answer"})

    graph.set_finish_point("answer")
    graph.set_finish_point("audit_synthesis")

    return graph.compile()
