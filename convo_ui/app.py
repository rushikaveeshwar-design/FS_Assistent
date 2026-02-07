import sys
import os
import streamlit as st
import uuid
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from memory.chat_store import ChatStore
from agent.tools import run_agent_stream
from memory.chat_store import generate_chat_title
from convo_ui.app_context import (llm, embedding_model, 
                                  clip_embedding_model, 
                                  vectorstore_manager)

# Page config
st.set_page_config(layout="wide",
                   page_title="Formula Student Engineering Assistant")

# Persist store
chat_store = ChatStore()

def make_agent_runner(*, llm, embedding_model, clip_embedding_model,
                      vectorstore_manager, chat_store):
    def run_agent(*, question, mode, competition, year, chat_id, user_images=None):
        return run_agent_stream(question=question, mode=mode,
                                competition=competition, year=year,
                                chat_id=chat_id, llm=llm,
                                embedding_model=embedding_model,
                                clip_embedding_model=clip_embedding_model,
                                vectorstore_manager=vectorstore_manager,
                                chat_store=chat_store, user_images=user_images or [])
    return run_agent

agent_runner = make_agent_runner(
    llm=llm,
    embedding_model=embedding_model,
    clip_embedding_model=clip_embedding_model,
    vectorstore_manager=vectorstore_manager,
    chat_store=chat_store,
)

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
    }

if "last_inspection_question" not in st.session_state:
    st.session_state.last_inspection_question = None

if "last_inspection_images" not in st.session_state:
    st.session_state.last_inspection_images = []

# Sidebar control panel
st.sidebar.title("Formula Student Assistant")
if st.sidebar.button("New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())

st.sidebar.markdown("### Your chats")

for cid, title in chat_store.list_chats():
    if st.sidebar.button(title, key=f"chat_{cid}"):
        st.session_state.chat_id = cid
        st.session_state.inspection_active = False
        st.session_state.last_inspection_question = None
        st.session_state.last_inspection_images = []
        st.rerun()

st.sidebar.markdown("### Mode")
st.session_state.mode = st.sidebar.radio("What's your assistance preference",
                                         ["Rule Q&A", "Design Audit", "Tech Inspection", "Comparison"])

# st.sidebar.markdown("---")
st.sidebar.markdown("### Event Context")
st.session_state.competition = st.sidebar.selectbox("competition",
                                                    ["General", "Formula Bharat", "FS Germany", "FSAE Supra", "Formula Imperial"],
                                                    index=0)

if st.session_state.competition == "General":
    st.session_state.competition = None

st.session_state.year = st.sidebar.selectbox("Rulebook Year",
                                             ["General", 2025, 2024, 2023, 2022, 2021],
                                             index=0)

if st.session_state.year == "General":
    st.session_state.year = None

st.sidebar.markdown("### Active Context")

st.sidebar.markdown(f"""
<div style="line-height:1.6">
                    <b>Mode</b>: <span style="color:#4CAF50">{st.session_state.mode}</span><br>
                    <b>Year</b>: <span style="color:#FF9800">{st.session_state.year or "Latest"}</span><br>
                    <b>Subsystem</b>: <span style="color:#9C27B0">{st.session_state.subsystem or "General"}</span>
</div>
""", unsafe_allow_html=True)

# Render previous chat messages
history = chat_store.get_messages(st.session_state.chat_id)

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Agent invocation (Streaming)
def stream_answer(generator):
    placeholder = st.empty()
    accumulated = ""

    for token in generator:
        accumulated += token
        placeholder.markdown(accumulated)

    return accumulated

def run_inspection_step(question):
    final_payload = {}

    def agent_stream():
        for event in agent_runner(
            question=question,
            mode="Tech Inspection",
            competition=st.session_state.competition,
            year=st.session_state.year,
            chat_id=st.session_state.chat_id,
            user_images=st.session_state.user_images,
        ):
            if "token" in event:
                yield event["token"]

            if "answer" in event:
                final_payload.update(event)

    answer_text = stream_answer(agent_stream())
    return answer_text, final_payload

# Tech inspection part
if st.session_state.mode == "Tech Inspection" and st.session_state.inspection_active:
    stage = st.session_state.inspection_state["inspection_stage"] + 1
    strictness = st.session_state.inspection_state["inspection_strictness"]


    with st.chat_message("FS_assistant"):
        st.markdown(f"Tech Inspection Questions and Progress: {stage} / 5")
        st.markdown(st.session_state.last_inspection_question)
    
    level_map = {0: ("Exploratory", 0.25),
                 1: ("Clarifying", 0.5),
                 2: ("Compliance-focused", 0.75),
                 3: ("Scrutineering-level", 1.0)}
    
    label, progress = level_map.get(strictness, ("Unknown", 0.0))

    st.markdown("#### Inspection Strictness")
    st.progress(progress)
    st.caption(f"level:**{label}**")

    if st.session_state.last_inspection_images:
        with st.expander("Reference Diagram"):
            for img in st.session_state.last_inspection_images:
                st.markdown(f"- **{img['source']}**, page {img['page']}")

# User input
user_input = st.chat_input("Describe your design" if st.session_state.mode=="Design Audit"
                           else "Ask anything within the event and engineering space")


if user_input:
    # If first chat
    if not chat_store.list_chats() or st.session_state.chat_id not in [
        cid for cid, _ in chat_store.list_chats()
    ]:
        title = generate_chat_title(user_input, st.session_state.mode)
        chat_store.create_chat(st.session_state.chat_id, title)

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

            question, payload = run_inspection_step(question=st.session_state.last_inspection_question)
            st.session_state.last_inspection_question = question
            st.session_state.last_inspection_images = payload.get("images", [])
            if "inspection_status" in payload:
                st.session_state.inspection_state["inspection_status"] = payload["inspection_status"]
            
            status = st.session_state.inspection_state.get("inspection_status")

            if status in ("PASS", "FAIL"):
                with st.chat_message("FS_assistant"):
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

    uploaded_files = st.file_uploader("Attach image file(s)",
                                      type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")
    if uploaded_files:
        if "user_images" not in st.session_state:
            st.session_state.user_images = []

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            st.session_state.user_images.append(image)
    
    if st.session_state.user_images:
        with st.expander("Uploaded images"):
            for img in st.session_state.user_images:
                st.image(img, use_container_width=True)

    with st.chat_message("FS_assistant"):

        final_payload = {}

        def agent_stream():
            for event in agent_runner(
                question=user_input,
                mode=st.session_state.mode,
                competition=st.session_state.competition,
                year=st.session_state.year,
                chat_id=st.session_state.chat_id,
                user_images=st.session_state.user_images
            ):
                if "token" in event:
                    yield event["token"]

                if "answer" in event:
                    final_payload.update(event)
        
        answer_text = stream_answer(agent_stream())

        chat_store.add_messages(st.session_state.chat_id,
                                "FS_assistant", answer_text)
        
        chat_store.touch_chat(st.session_state.chat_id)

        # Design audit rendering
        if st.session_state.mode == "Design Audit" and "audit" in final_payload:
            st.markdown("### Design Audit Results")

            for item in final_payload["audit"]:
                st.markdown(f"**Claim:** {item['claim']}")
                st.markdown(f"**Status:** {item['status']}")
                st.markdown(f"**Reason:** {item['reason']}")

                if item.get("citations"):
                    with st.expander("Rule References"):
                        for citation in item["citations"]:
                            st.markdown(f"- Section {citation['section']} ({citation['confidence']})")

        if final_payload.get("images"):
            with st.expander("Relevant diagrams and figures"):
                for img in final_payload["images"]:
                    st.markdown(f"- **{img['source']}**, page {img['page']}"
                                f"({img.get('diagram')})")
