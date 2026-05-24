from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatMistralAI(name="mistral-small-2506", temperature=0.9)

# list typed as BaseMessage so we can append HumanMessage and AIMessage into it
# SystemMessage at index 0 sets the AI persona for the whole conversation
messages: list[BaseMessage] = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    print("-" * 50)
    print("Type 'exit' to quit.")
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    # keep adding to messages so the model remembers the full conversation
    messages.append(AIMessage(content=response.content))
    print(f"Mistral: {response.content}")
