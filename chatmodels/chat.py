from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

load_dotenv()

# two ways to init a model in langchain:
# 1. init_chat_model — one function, works for any provider
# 2. provider class directly — more explicit, more control

# --- init_chat_model ---
# format is "provider:model-name"

# model = init_chat_model("google_genai:gemini-2.5-flash-lite")
# response = model.invoke("Who is no. 1 T20i batsman?")
# print(response.content)

# model = init_chat_model("groq:llama-3.1-8b-instant")
# response = model.invoke("What is RAG? Tell in short.")
# print(response.content)

# --- provider classes ---

# Gemini
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# response = model.invoke("What is RAG?")
# print(response.content)

# Groq — very fast inference
# model = ChatGroq(model="llama-3.1-8b-instant")
# response = model.invoke("What is RAG? Tell in short.")
# print(response.content)

# Mistral
# model = ChatMistralAI(name="mistral-small-2506")
# response = model.invoke("What is RAG? Tell in short.")
# print(response.content)

# temperature — how creative/random the output is (0 = focused, 1 = creative)
# max_tokens — hard limit on how long the response can be
model = ChatMistralAI(name="mistral-small-2506", temperature=0.9, max_tokens=20)
response = model.invoke("Write a haiku about the ocean.")
print(response.content)
