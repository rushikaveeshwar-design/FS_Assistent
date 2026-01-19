import streamlit as st
import uuid
from memory.chat_store import ChatStore

st.set_page_config(lauout="wide")

chat_store = ChatStore()

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

st.sidebar.title("Formula Student Assistant")

if st.sidebar.button("New Chat"):
    st.session_state.chat_id = str(uuid.uuid4())

user_input = st.text_input("ask your question")

if user_input:
    chat_store.add_messsage(st.session_state.chat_id, "user", user_input)

    # agent.invoke() comes here
    response = "Agent response placeholder"

    chat_store.add_message(st.session_state.chat_id, "assistant", response)

    st.write(response)

with st.expander("Source & Citations"):
    for c in response["citations"]:
        st.markdown(f"- **{c['competition']} {c['year']} | {c['section']}** "
                    f"({c['confidence']})")
    