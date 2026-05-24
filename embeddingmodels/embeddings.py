from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
)
vector = embeddings.embed_query("What is the capital of France?")
print(vector)