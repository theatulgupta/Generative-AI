from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatMistralAI(name = "mistral-small-2506", temperature=0.9)

# Typed as list[BaseMessage] so Pylance accepts HumanMessage and AIMessage appends
messages: list[BaseMessage] = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    print("-" * 50)
    print("Type 'exit' to quit.")
    prompt = input("You: ")

    # Check exit before appending to avoid adding it to history
    if prompt.lower() == "exit":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    # Append AI response to maintain full conversation history for next turn
    messages.append(AIMessage(content=response.content))
    print(f"Mistral: {response.content}")
