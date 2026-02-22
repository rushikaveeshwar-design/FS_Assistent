from retrieval.rule_retriever import retrieve_rules
from retrieval.engg_retriever import retrieve_engg
from typing import Generator, Dict, Literal, List, Any
from agent.main import build_graph, ChatState
from agent.logger import log_event, log_exception
from memory.user_memory_engine import UserMemoryEngine
import time

class AgentTool:
    def __init__(self, rule_store=None, engg_store=None, image_store=None,
                 vectorstore_manager=None, chat_store=None):
        self.rule_store = rule_store
        self.engg_store = engg_store
        self.image_store = image_store
        self.vectorstore_manager = vectorstore_manager
        self.chat_store = chat_store
        self.memory_engine = UserMemoryEngine(chat_store)

    def get_rules(self, query, competition, year, domain=None):
        results = []
        for store in self.rule_store:
            results.extend(retrieve_rules(store=store, query=query,
                                          competition=competition, year=year,
                                          domain=domain))
        return results
    
    def get_engg(self, query):
        results = []
        for store in self.engg_store:
            results.extend(retrieve_engg(store=self.engg_store, query=query))
        return results
    
    def get_image_by_vector(self, query_vector, k=1):
        if not self.image_store:
            return []
        results = []
        for store in self.image_store:
            results.extend(store.similarity_search_by_vector(query_vector, k=k))

def compile_graph(llm, vectorstore_manager, chat_store, clip_embedding_model):
    
    log_event("INFO", "compile_graph_start")

    tools = AgentTool(vectorstore_manager=vectorstore_manager, 
                      chat_store=chat_store)
    graph = build_graph(llm, tools, clip_model=clip_embedding_model)

    log_event("INFO", "compile_graph_done")
    return graph, tools
    
def run_agent_stream(*, question: str, mode: str,
                     competition: str | None, year: int | None,
                     chat_id: str, llm, clip_embedding_model,
                     vectorstore_manager, chat_store, user_images) -> Generator[Dict, None, None]:
    """Yields incremental agent output.
    
    Final yield MUST contain:
    {"answer": str, "assumptions": list[str],
    "citations": list[dict], "images": list[dict] | None}

    """
    log_event("INFO", "agent_run_start", chat_id=chat_id,
              meta={"mode": mode, "competition": competition, "year": year})
    
    t0_total = time.time()
    
    try:

        compiled_graph, tools = compile_graph(llm, vectorstore_manager=vectorstore_manager, 
                                              chat_store=chat_store, clip_embedding_model=clip_embedding_model)
        
        memory_engine = tools.memory_engine

        # Initial state
        state = ChatState(question=question, mode=mode, subqueries=[],
                          competition=competition, year=year,
                          retrieved_rules=[], retrieved_engg=[],
                          intent=None, user_images=user_images or [],
                          image_observations=[])
        state.chat_id = chat_id

        # streaming
        for update in compiled_graph.stream(state):
            # Only have to stream answer tokens
            if "answer_token" in update:
                yield {"token": update["answer_token"]}

            if "final_answer" in update:
                log_event("INFO", "agent_run_complete", chat_id=chat_id)

                total_latency = int((time.time() - t0_total)*1000)

                log_event("INFO", "agent_run_complete",
                          chat_id=chat_id, meta={"mode": mode,
                                                 "latency": total_latency})

                final_payload = update["final_answer"]                

                if "images" not in final_payload:
                    final_payload["images"] = []
                
                memory_engine.update(chat_id, llm, new_info=f"""
                                    User question: {question}
                                    Assistent answer: {final_payload["answer"]}""")

                yield final_payload
    
    except Exception as e:
        total_latency = int((time.time() - t0_total)*1000)

        log_event("ERROR", "agent_run_failed", chat_id=chat_id,
                  meta={"mode": mode, "latency": total_latency})

        log_exception(e, chat_id=chat_id, node="run_agent_stream")
        raise

DiagramMode = Literal["none", "doc_reference", 
                      "user_refernce", "comparitative"]

def extract_visual_observations(*, state: ChatState, user_images: List[Any], 
                                vision_llm) -> List[Dict[str, str]]:
    """
    Convert uploaded images into structured, uncertainty-aware observations.
    This does not judge compliance.
    """

    observations = []

    for idx, image in enumerate(user_images):
        prompt = f"""
You are analyzing an engineering related image for a Formula Student design and conceptulization.

Task:
- Describe ONLY what is directly visible.
- Do not infer hidden structure dimensions and compliance.
- If something is unclear, explicitly say so.

Return:
- Understanding of the image regarding to the query inputed by the user,
by answering the question.
- The Key components visible.
- Possible compliance relevant features.
- Any uncertainty.
- Return output strictly as JSON:
{{
    "image_index": {idx},
    "observations": "...",
    "uncertainties": "..."
}}
"""
        try:
            t0 = time.time()

            result = vision_llm.invoke(prompt, image=image)

            latency = int((time.time() - t0)*1000)

            log_event("DEBUG", "vision_llm_invoke", chat_id=state.chat_id,
                      meta={"image_index": idx,
                            "latency": latency})

            observations.append(result)
            
        except Exception as e:
            log_exception(e, chat_id=state.chat_id, 
                          node="extract_visual_observations")

            observations.append({
                "image_index": idx,
                "observations": "Unable to analyze image",
                "uncertainties": str(e)
            })
        
    return observations

def infer_diagram_relevance(*, mode: str, question: str, 
                            retrieved_rules: List[Any],
                            retrieved_engg: List[Any],
                            images: List[dict]) -> DiagramMode:
    """Decide how diagram should be used for this interaction.
    This function does not retrieve images - only decides reasoning strategy."""

    q = question.lower()

    diagram_triggers = ["diagram", "figure", "layout", "schematic",
                        "drawing", "arrangement", "geometry",
                        "shown", "illustrated", "depicted"]
    
    user_visual_triggers = ["my design", "this setup", "attached image",
                            "shown here", "as you can see", "my layout"]
    
    rule_diagrams_present = bool(images)
    mentions_diagram = any(trigger in q for trigger in diagram_triggers)
    mentions_user_visual = any(trigger in q for trigger in user_visual_triggers)

    if mentions_user_visual and rule_diagrams_present:
        return "comparitative"
    
    if mentions_user_visual:
        return "user_reference"
    
    if mentions_diagram or rule_diagrams_present:
        return "doc_reference"
    
    return "none"

def build_diagram_instructions(*, diagram_mode: DiagramMode,
                               images: List[dict]):
    """Injects *precise* instructions into the prompt based on diagram usage."""

    if diagram_mode == "none":
        return ""
    
    if diagram_mode == "doc_reference":
        return f"""
Diagram handling:
- The following diagrams may be relevant:
{images}
- Use them ONLY as explainatory or clarifying reference.
- Do not assume dimenstions, coverage, or compliance unless explicitly stated.
"""
    if diagram_mode == "user_reference":
        return """
User image handling:
- User-submitted images are observational only.
- Use them to support or question claims.
- If image clarity is insufficient, explicitly state uncertainty.
"""
    if diagram_mode == "comparitative":
        return f"""
Comparitative  diagram reasoning:
- Compare user-submitted images against the following official diagrams:
{images}
- Identify matches, mismatches and unclear areas.
- Do not assume equivalence unless explicitly visible.
- If a diagram implies a requirement not visible or clarified in the image,
mark it as ambiguous and request clarification. 
"""
    return ""

