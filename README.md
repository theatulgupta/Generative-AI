# 🤖 Generative AI with LangChain

A hands-on collection of LangChain concepts and projects built with Python — covering chat models, embeddings, runnables, and real-world AI apps.

---

## 📁 Project Structure

```
GenAI/
├── chatmodels/          # Chat model integrations & chatbot UIs
├── embeddingmodels/     # Text embedding with OpenAI & HuggingFace
├── runnables/           # LangChain LCEL runnable patterns
├── cinesage/            # 🎬 Movie info extraction app
├── requirements.txt
└── README.md
```

---

## 🧩 Modules

### 💬 Chat Models — `chatmodels/`

Exploring different LLM providers using LangChain's unified interface.

| File | Description |
|------|-------------|
| `chat.py` | Gemini, Groq & Mistral via `init_chat_model` and model classes |
| `chatbot.py` | Terminal chatbot with conversation history using Mistral |
| `huggingface.py` | Chat with DeepSeek via HuggingFace Inference Endpoint |
| `localmodel.py` | Run TinyLlama locally using `HuggingFacePipeline` |
| `UIChatBot.py` | Streamlit chatbot UI with mood selection (Helpful / Funny / Strict) |

---

### 🔢 Embedding Models — `embeddingmodels/`

Converting text into vector representations for semantic search and RAG.

| File | Description |
|------|-------------|
| `embeddings.py` | OpenAI `text-embedding-3-small` with 1024 dimensions |
| `hf_embeddings.py` | HuggingFace `all-MiniLM-L6-v2` sentence embeddings |

---

### ⛓️ Runnables (LCEL) — `runnables/`

LangChain Expression Language (LCEL) patterns for composing chains.

| File | Description |
|------|-------------|
| `sequence_runnables.py` | `prompt \| model \| parser` — basic sequential chain |
| `parallel_runnables.py` | `RunnableParallel` — run multiple chains simultaneously |
| `passthrough_runnables.py` | `RunnablePassthrough` — pass input alongside chain output |

---

### 🎬 CineSage — `cinesage/`

A Streamlit app that extracts structured movie information from a paragraph using Mistral AI.

| File | Description |
|------|-------------|
| `core.py` | Plain text extraction using `StrOutputParser` |
| `core_pydantic.py` | Structured JSON extraction using `PydanticOutputParser` |

**Output fields:** Title, Genre, Director, Cast, Rating, Plot Summary, Themes, Notable Features

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral_AI-mistral--small--2506-FF7000?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)

- **LangChain** — LCEL, Runnables, Prompt Templates, Output Parsers
- **Mistral AI** — `mistral-small-2506` as primary LLM
- **Google Gemini** — via `langchain-google-genai`
- **Groq** — fast inference with `llama-3.1-8b-instant`
- **HuggingFace** — cloud endpoints & local model pipelines
- **Streamlit** — interactive web UIs
- **Pydantic** — structured output validation

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/theatulgupta/Generative-AI.git
cd Generative-AI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Add your API keys inside .env
```

```env
MISTRAL_API_KEY=<your_mistral_api_key>
GOOGLE_API_KEY=<your_google_api_key>
GROQ_API_KEY=<your_groq_api_key>
OPENAI_API_KEY=<your_openai_api_key>
HUGGINGFACEHUB_ACCESS_TOKEN=<your_hf_token>
```

**4. Run any script**
```bash
python chatmodels/chatbot.py
streamlit run cinesage/core_pydantic.py
```

---

## 📚 Concepts Covered

- ✅ LLM initialization with multiple providers
- ✅ Prompt Templates (`ChatPromptTemplate`)
- ✅ Output Parsers (`StrOutputParser`, `PydanticOutputParser`)
- ✅ LCEL — Sequential, Parallel & Passthrough Runnables
- ✅ Conversation history with `SystemMessage`, `HumanMessage`, `AIMessage`
- ✅ Text Embeddings (OpenAI + HuggingFace)
- ✅ Structured output with Pydantic schemas
- ✅ Streamlit UI integration

---

<p align="center">Made with ❤️ while learning LangChain</p>
