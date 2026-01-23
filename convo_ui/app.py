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

submit = st.button("Submit")

# Agent invocation (Streaming)
def stream_answer(generator):
    placeholder = st.empty()
    accumulated = ""

    for token in generator:
        accumulated += token
        placeholder.markdown(accumulated)

    return accumulated


if submit and user_input.strip():
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