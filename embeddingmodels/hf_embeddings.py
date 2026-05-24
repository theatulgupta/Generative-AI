from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# all-MiniLM-L6-v2 is a lightweight sentence transformer — outputs 384-dim vectors
# runs locally, no API cost, great for semantic similarity tasks
# note: HuggingFaceEmbeddings doesn't take a dimensions= param, size is fixed by the model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vector = embeddings.embed_query("What is the capital of France?")
print(f"Vector length: {len(vector)}")
print(vector)
