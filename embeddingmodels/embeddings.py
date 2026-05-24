from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# text-embedding-3-small default is 1536 dims, we reduce to 1024 to save memory
# dimensions param is only supported by OpenAI's v3 models
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
)

vector = embeddings.embed_query("What is the capital of France?")
print(f"Vector length: {len(vector)}")
print(vector)
