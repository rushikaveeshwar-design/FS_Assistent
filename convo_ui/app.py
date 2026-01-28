import streamlit as st
import uuid
from memory.chat_store import ChatStore
from agent.tools import run_agent_stream

# Page config
st.set_page_config(lauout="wide",
                   page_title="Formula Student Engineering Assistant")

# Persist store
chat_store = ChatStore()

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "mode" not in st.session_state:
    st.session_state.mode = "Rule Q&A"

if "competition" not in st.session_state:
    st.session_state.competition = None

if "year" not in st.session_state:
    st.session_state.year = None

if "subsystem" not in st.session_state:
    st.session_state.subsystem = None

if "inspection_active" not in st.session_state:
    st.session_state.inspection_active = False

if "inspection_state" not in st.session_state:
    st.session_state.inspection_state = {
        "inspection_stage": 0,
        "inspection_strictness": 0,
        "inspection_status": None,
        "inspection_history": [],
        "last_user_answer": None
    }

if "last_inspection_question" not in st.session_state:
    st.session_state.last_inspection_question = None

if "last_inspection_images" not in st.session_state:
    st.session_state.last_inspection_images = []

# Sidebar control panel
# st.sidebar.title("Formula Student Assistant")
if st.sidebar.button("New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())

st.sidebar.markdown("### Mode")
st.session_state.mode = st.sidebar.radio("What's your assistance preference",
                                         ["Rule Q&A", "Design Audit", "Tech Inspection", "Comparison"])

# st.sidebar.markdown("---")
st.sidebar.markdown("### Event Context")
st.session_state.competition = st.sidebar.selectbox("competition",
                                                    ["General", "Formula Bharat", "FS Germany", "FSAE Supra", "Formula Imperial"])

if st.session_state.competition == "General":
    st.session_state.competition = None

st.session_state.year = st.sidebar.selectbox("Rulebook Year",
                                             ["General", 2025, 2024, 2023, 2022, 2021])

if st.session_state.year == "General":
    st.session_state.year = None

# Header Visible system state
st.markdown(f"""
**Mode:** {st.session_state.mode}
**competition:** {st.session_state.competition or "General"}
**Year:** {st.session_state.year or "latest"}
**Subsystem:** {st.session_state.subsystem or "General"}
""")

# Input section
if st.session_state.mode == "Design Audit":
    user_input = st.text_area("Describe your design",
                              height=200,
                              placeholder="Describe your design in clear technical terms")
else:
    user_input = st.text_input("Ask your question",
                               placeholder="Ask anything within the event space")

if st.session_state.mode == "Tech Inspection" and st.session_state.inspection_active:
    st.markdown("### Current Inspection Question")
    st.markdown(st.session_state.last_inspection_question)

if (st.session_state.mode == "Tech Inspection"
    and st.session_state.inspection_active
    and st.session_state.get("last_inspection_images")):
    with st.expander(" Referenced Diagram"):
        for img in st.session_state.last_inspection_images:
            st.markdown(f"- **{img["source"]}**, page {img["page"]}")


submit = st.button("Submit")

# Agent invocation (Streaming)
def stream_answer(generator):
    placeholder = st.empty()
    accumulated = ""

    for token in generator:
        accumulated += token
        placeholder.markdown(accumulated)

    return accumulated

def run_inspection_step(question, last_user_answer=None):
    final_payload = {}

    def agent_stream():
        for event in run_agent_stream(question=question,
                                      mode="Tech Inspection",
                                      competition=st.session_state.competition,
                                      year=st.session_state.year,
                                      chat_id=st.session_state.chat_id,
                                      **st.session_state.inspection_state,
                                      last_user_answer=last_user_answer):
            if "token" in event:
                yield event["token"]

            if "answer" in event:
                final_payload.update(event)
        
    answer_text = stream_answer(agent_stream())
    return answer_text, final_payload

if submit and user_input.strip():

    if st.session_state.mode == "Tech Inspection":
        if not st.session_state.inspection_active:
            st.session_state.inspection_active = True

            question, payload = run_inspection_step(question=user_input)
            st.session_state.last_inspection_question = question
            st.session_state.last_inspection_images = payload.get("images", [])
            st.session_state.inspection_state["inspection_history"].append({"question": question})

            st.stop()
    
        else:
            user_answer = user_input
            st.session_state.inspection_state["last_user_answer"] = user_answer

            question, payload = run_inspection_step(question=st.session_state.last_inspection_question,
                                                    last_user_answer=user_answer)
            st.session_state.last_inspection_question = question
            st.session_state.last_inspection_images = payload.get("images", [])
            if "inspection_status" in payload:
                st.session_state.inspection_state["inspection_status"] = payload["inspection_status"]
            
            status = st.session_state.inspection_state.get("inspection_status")

            if status in ("PASS", "FAIL"):
                st.markdown(f"### Inspection Result: {status}")
                st.markdown(payload["answer"])
            
            # reset inspection state
                st.session_state.inspection_active = False
                st.session_state.last_inspection_question = None
                st.session_state.inspection_state = {
                    "inspection_stage": 0,
                    "inspection_strictness": 0,
                    "inspection_status": None,
                    "inspection_history": [],
                    "last_user_answer": None
                }
                st.session_state.last_inspection_images = []
                st.stop()
            
            # continue inspection
            st.session_state.last_inspection_question = question
            st.session_state.inspection_state["inspection_history"].append({"question": question,
                                                                            "user_answer": user_answer})
            st.stop()

    # Storing user message
    chat_store.add_message(
        st.session_state.chat_id,
        "user",
        user_input,
    )

    with st.status("Processing request...", expanded=True) as status:
        st.write("Interpreting intent")
        st.write("Retrieving relevant rules")

        final_payload = {}

        def agent_stream():
            for event in run_agent_stream(
                question=user_input,
                mode=st.session_state.mode,
                competition=st.session_state.competition,
                year=st.session_state.year,
                chat_id=st.session_state.chat_id,
            ):
                if "token" in event:
                    yield event["token"]

                if "answer" in event:
                    final_payload.update(event)

        response = {
            "answer": stream_answer(agent_stream()),
            "assumptions": final_payload.get("assumptions", []),
            "citations": final_payload.get("citations", []),
        }

        if st.session_state.mode == "Design Audit" and "audit" in final_payload:
            st.markdown("### Design Audit Results")

            for item in final_payload["audit"]:
                st.markdown(f"**Claim:** {item["claim"]}")
                st.markdown(f"**Status:** {item["status"]}")
                st.markdown(f"**Reason:** {item["reason"]}")

                if item.get("citations"):
                    with st.expander("Rule References"):
                        for citation in item["citations"]:
                            st.markdown(f"- Section {citation["section"]} ({citation["confidence"]})")

        if response.get("images"):
            with st.expander("Relevant diagrams and figures"):
                for img in response["images"]:
                    st.markdown(f"- **{img["source"]}**, page {img["page"]}"
                                f"({img.get("caption, diagram")})")