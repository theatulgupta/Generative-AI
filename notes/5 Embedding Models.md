# 5. Embedding Models

Embeddings convert text into:

```text
Dense numerical vectors
```

that capture semantic meaning.

```text
"I love cricket"  →  [0.23, -0.11, 0.87, ...]
"I enjoy cricket" →  [0.24, -0.10, 0.85, ...]  ← similar!
"I hate broccoli" →  [-0.91, 0.44, -0.32, ...]  ← different
```

Similar meaning → similar vectors (close in vector space).

---

## Why Embeddings?

Used in:

- semantic search
- RAG (Retrieval Augmented Generation)
- document similarity
- clustering
- recommendation systems

---

## embed_query vs embed_documents

```python
vector = embeddings.embed_query("What is the capital of France?")

# Multiple texts
vectors = embeddings.embed_documents(["text1", "text2", "text3"])
```

---

## 1. OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
)

vector = embeddings.embed_query("What is the capital of France?")
print(len(vector))  # 1024
```

Needs: `OPENAI_API_KEY`

Model: `text-embedding-3-small`

- dimensions: up to 1536 (can reduce with `dimensions` param)
- good balance of quality and cost

---

## 2. HuggingFace Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vector = embeddings.embed_query("What is the capital of France?")
print(len(vector))  # 384
```

Needs: `HUGGINGFACEHUB_ACCESS_TOKEN`

- FAISS (local, Facebook)
- Pinecone (cloud)
- Chroma (local)
- Weaviate (cloud/local)

Model: `all-MiniLM-L6-v2`

- 384 dimensions
- fast and lightweight
- great for semantic similarity tasks
- runs locally (no API cost)

---

## Embedding Dimensions

| Model                    | Dimensions          | Notes                      |
| ------------------------ | ------------------- | -------------------------- |
| `text-embedding-3-small` | 1024 (configurable) | OpenAI, paid               |
| `text-embedding-3-large` | 3072                | OpenAI, more accurate      |
| `all-MiniLM-L6-v2`       | 384                 | HuggingFace, free, fast    |
| `all-mpnet-base-v2`      | 768                 | HuggingFace, more accurate |

Higher dimensions:

```text
More accurate but slower and more memory
```

---

## Cosine Similarity

How to measure if two vectors are similar:

$$
\text{similarity} = \cos(\theta) = \frac{A \cdot B}{|A||B|}
$$

```text
similarity = 1   → identical meaning
similarity = 0   → unrelated
similarity = -1  → opposite meaning
```

---

## Vector Database (Concept)

Embeddings are stored in a vector database for fast similarity search.

```text
Query → embed_query → vector → search DB → top-k similar docs
```

Popular vector DBs:

- FAISS (local, Facebook)
- Pinecone (cloud)
- Chroma (local)
- Weaviate (cloud/local)

FAISS is in `requirements.txt`:

```text
faiss-cpu
```

---

## RAG Overview (Why Embeddings Matter)

```text
1. Load documents
2. Split into chunks
3. Embed each chunk → store in vector DB
4. User asks question → embed question
5. Find similar chunks (cosine similarity)
6. Pass chunks + question to LLM
7. LLM answers using retrieved context
```

This solves the LLM limitation of:

```text
No access to your private/real-time data
```

---

## Important Interview Points

- embeddings = dense vectors capturing semantic meaning
- similar text → similar vectors (close in space)
- `embed_query` for single text, `embed_documents` for batch
- OpenAI embeddings are paid, HuggingFace can run locally
- dimensions = size of the vector (higher = more info, more cost)
- cosine similarity measures how close two vectors are
- embeddings are the foundation of RAG and semantic search
- FAISS is a popular local vector store for similarity search
