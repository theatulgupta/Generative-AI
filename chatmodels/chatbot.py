from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Using Temperature and Max Tokens parameters with Mistral
model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

messages = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    print("Type 'exit' to quit.")
    print("-" * 50)
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower() == "exit":
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f'Mistral: {response.content}')
