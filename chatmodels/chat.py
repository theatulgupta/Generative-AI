from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

# Initialize model using init_chat_model - Gemini
'''
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
response = model.invoke("Who is no. 1 T20i batsman?")
print(response)
'''

from langchain_google_genai import ChatGoogleGenerativeAI
# Initialize using Model Class - Gemini
'''
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
response = model.invoke("What is RAG?")
print(response.content)
'''


# Initialize model using init_chat_model - Groq Model
'''
model = init_chat_model("groq:llama-3.1-8b-instant")
response = model.invoke("What Retrieval Augmented Generation? Tell in short.")
print(response.content)
'''

from langchain_groq import ChatGroq
# Initialize using Model Class - Groq
'''
model = ChatGroq(model="llama-3.1-8b-instant")
response = model.invoke("What Retrieval Augmented Generation? Tell in short.")
print(response.content)
'''

from langchain_mistralai import ChatMistralAI
# Initialize using Model Class - Mistral
'''
model = ChatMistralAI(model="mistral-small-2506")
response = model.invoke("What Retrieval Augmented Generation? Tell in short.")
print(response.content)
'''

# Using Temperature and Max Tokens parameters with Mistral
model = ChatMistralAI(model="mistral-small-2506", temperature=0.9, max_tokens=20)
response = model.invoke("Write a haiku about the ocean.")
print(response.content)