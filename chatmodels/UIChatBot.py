import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load env
load_dotenv()

# Initialize model
model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

st.set_page_config(page_title="Mistral Chatbot", page_icon="🤖")

st.title("🤖 Mistral Chatbot")
mood = st.selectbox(
    "Select Mood",
    ["Helpful 😊", "Funny 😂", "Strict 🧑‍🏫"]
)

# Initialize chat history
if "messages" not in st.session_state:
    system_prompt = "You are a helpful assistant."

    if mood == "Funny 😂":
        system_prompt = "You are a funny assistant who replies with humor."
    elif mood == "Strict 🧑‍🏫":
        system_prompt = "You are a strict assistant who gives concise and direct answers."

    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Update system message if mood changes
system_prompt = "You are a helpful assistant."
if mood == "Funny 😂":
    system_prompt = "You are a funny assistant who replies with humor."
elif mood == "Strict 🧑‍🏫":
    system_prompt = "You are a strict assistant who gives concise and direct answers."

if isinstance(st.session_state.messages[0], SystemMessage):
    st.session_state.messages[0] = SystemMessage(content=system_prompt)

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    response = model.invoke(st.session_state.messages)
    
    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    
    with st.chat_message("assistant"):
        st.markdown(response.content)