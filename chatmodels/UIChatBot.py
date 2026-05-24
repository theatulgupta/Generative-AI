import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

st.set_page_config(page_title="Mistral Chatbot", page_icon="🤖")
st.title("🤖 Mistral Chatbot")

mood = st.selectbox("Select Mood", ["Helpful 😊", "Funny 😂", "Strict 🧑🏫"])


def get_system_prompt(mood: str) -> str:
    """Return system prompt based on selected mood."""
    if mood == "Funny 😂":
        return "You are a funny assistant who replies with humor."
    elif mood == "Strict 🧑🏫":
        return "You are a strict assistant who gives concise and direct answers."
    return "You are a helpful assistant."


# Initialize chat history with system message on first load
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=get_system_prompt(mood))]

# Update system message whenever mood changes
st.session_state.messages[0] = SystemMessage(content=get_system_prompt(mood))

# Display chat history — skip the system message at index 0
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle new user input
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass full message history so the model has conversation context
    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.markdown(response.content)
