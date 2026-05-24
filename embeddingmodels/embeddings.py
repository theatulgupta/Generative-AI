from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# text-embedding-3-small supports custom dimensions (default 1536, here reduced to 1024)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
)

vector = embeddings.embed_query("What is the capital of France?")
print(f"Vector length: {len(vector)}")
print(vector)