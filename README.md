# 🤖 Generative AI with LangChain

A hands-on collection of LangChain concepts and projects built with Python — covering chat models, embeddings, runnables, tools, agents, and real-world AI apps.

---

## 📁 Project Structure

```
GenAI/
├── chatmodels/          # Chat model integrations & chatbot UIs
├── embeddingmodels/     # Text embedding with OpenAI & HuggingFace
├── runnables/           # LangChain LCEL runnable patterns
├── cinesage/            # 🎬 Movie info extraction app
├── tools/               # LangChain tools & tool calling
├── agents/              # LLM agents with manual & autonomous loops
├── notes/               # 📝 Revision notes for every concept
├── .env.example         # Required API keys template
├── requirements.txt
└── README.md
```

---

## 🧩 Modules

### 💬 Chat Models — `chatmodels/`

| File | Description |
|------|-------------|
| `chat.py` | Gemini, Groq & Mistral via `init_chat_model` and model classes |
| `chatbot.py` | Terminal chatbot with conversation history using Mistral |
| `huggingface.py` | Chat with DeepSeek via HuggingFace Inference Endpoint |
| `localmodel.py` | Run TinyLlama locally using `HuggingFacePipeline` |
| `UIChatBot.py` | Streamlit chatbot UI with mood selection (Helpful / Funny / Strict) |

---

### 🔢 Embedding Models — `embeddingmodels/`

| File | Description |
|------|-------------|
| `embeddings.py` | OpenAI `text-embedding-3-small` with 1024 dimensions |
| `hf_embeddings.py` | HuggingFace `all-MiniLM-L6-v2` sentence embeddings (384-dim) |

---

### ⛓️ Runnables (LCEL) — `runnables/`

| File | Description |
|------|-------------|
| `sequence_runnables.py` | `prompt \| model \| parser` — basic sequential chain |
| `parallel_runnables.py` | `RunnableParallel` + `RunnableLambda` — run multiple chains simultaneously |
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

### 🔧 Tools — `tools/`

| File | Description |
|------|-------------|
| `custom_tool.py` | Creating a custom tool with `@tool` decorator |
| `search_tool.py` | Real-time web search using `TavilySearch` in a chain |
| `tool_calling.py` | Manual tool calling flow: bind → invoke → ToolMessage → final answer |

---

### 🤖 Agents — `agents/`

| File | Description |
|------|-------------|
| `agents.py` | Manual agent loop with human-in-the-loop approval |
| `agents_autonomous.py` | Autonomous agent using `create_agent` |
| `agents_wrap_tool_call.py` | Agent with `wrap_tool_call` middleware for approval |

---

### 📝 Notes — `notes/`

Revision notes covering every concept studied.

| File | Topics |
|------|--------|
| `1 Intro to GenAI and LLMs.md` | GenAI, LLMs, tokens, temperature, context window |
| `2 Chat Models.md` | Providers, message types, conversation history, HuggingFace, Streamlit |
| `3 Prompt Templates and Output Parsers.md` | ChatPromptTemplate, StrOutputParser, PydanticOutputParser, Pydantic |
| `4 Runnables and LCEL.md` | Pipe operator, sequential/parallel/passthrough, RunnableLambda |
| `5 Embedding Models.md` | OpenAI & HuggingFace embeddings, cosine similarity, vector DBs, RAG |
| `6 Tools.md` | @tool decorator, bind_tools, tool_calls, ToolMessage, Tavily |
| `7 Agents.md` | Agent loop, ReAct pattern, human-in-the-loop, create_agent, middleware |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral_AI-mistral--small--latest-FF7000?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)

- **LangChain** — LCEL, Runnables, Prompt Templates, Output Parsers, Tools, Agents
- **Mistral AI** — `mistral-small-latest` as primary LLM
- **Google Gemini** — via `langchain-google-genai`
- **Groq** — fast inference with `llama-3.1-8b-instant`
- **HuggingFace** — cloud endpoints & local model pipelines
- **Tavily** — real-time web search for LLMs
- **OpenWeather API** — weather data for agent tools
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
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

**4. Run any script**
```bash
# Terminal scripts
python chatmodels/chatbot.py
python tools/tool_calling.py
python agents/agents.py

# Streamlit apps
streamlit run chatmodels/UIChatBot.py
streamlit run cinesage/core_pydantic.py
```

---

## 📚 Concepts Covered

- ✅ LLM initialization with multiple providers (Mistral, Gemini, Groq, HuggingFace)
- ✅ Prompt Templates (`ChatPromptTemplate`)
- ✅ Output Parsers (`StrOutputParser`, `PydanticOutputParser`)
- ✅ LCEL — Sequential, Parallel & Passthrough Runnables
- ✅ Conversation history with `SystemMessage`, `HumanMessage`, `AIMessage`
- ✅ Text Embeddings (OpenAI + HuggingFace)
- ✅ Structured output with Pydantic schemas
- ✅ Streamlit UI integration
- ✅ Custom tools with `@tool` decorator
- ✅ Tool calling flow — `bind_tools`, `tool_calls`, `ToolMessage`
- ✅ Real-time search with Tavily
- ✅ Manual agent loop with human-in-the-loop approval
- ✅ Autonomous agents with `create_agent`
- ✅ Tool call middleware with `wrap_tool_call`

---

<p align="center">Made with ❤️ while learning LangChain</p>
