# 2. Chat Models

Chat models are LLMs designed for:

```text
Conversational interaction
```

They take a list of messages as input and return a message as output.

Instead of:

```text
Single string in → Single string out
```

they work with:

```text
List of messages → AI message response
```

---

## LangChain Chat Model Interface

LangChain provides a unified interface to work with multiple LLM providers.

```text
Same code → Switch between Gemini, Groq, Mistral, OpenAI
```

Two ways to initialize:

---

### Method 1: `init_chat_model` (Universal)

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
model = init_chat_model("groq:llama-3.1-8b-instant")
```

Format:

```text
"provider:model-name"
```

---

### Method 2: Model Class (Explicit)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
model = ChatGroq(model="llama-3.1-8b-instant")
model = ChatMistralAI(name="mistral-small-2506")
```

---

## Invoking a Model

```python
response = model.invoke("What is RAG?")
print(response.content)
```

`.invoke()` returns an `AIMessage` object.

Access the text with `.content`.

---

## Model Parameters

```python
model = ChatMistralAI(
    name="mistral-small-2506",
    temperature=0.9,
    max_tokens=20
)
```

| Parameter | Purpose |
|-----------|---------|
| `temperature` | Controls randomness (0 = focused, 1 = creative) |
| `max_tokens` | Limits response length |

---

## Providers Used

### Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

Needs: `GOOGLE_API_KEY`

---

### Groq

```python
from langchain_groq import ChatGroq

model = ChatGroq(model="llama-3.1-8b-instant")
```

Needs: `GROQ_API_KEY`

Groq is known for very fast inference speed.

---

### Mistral AI

```python
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(name="mistral-small-2506")
```

Needs: `MISTRAL_API_KEY`

Used as primary model throughout this project.

---

## Message Types

LangChain uses structured message objects:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
```

| Message Type | Role | Purpose |
|---|---|---|
| `SystemMessage` | system | Sets AI behavior/persona |
| `HumanMessage` | user | User's input |
| `AIMessage` | assistant | AI's response |

---

## Chatbot with Conversation History

Problem:

```text
LLMs are stateless by default
```

Each call is independent — no memory.

Solution:

```text
Manually maintain a list of messages
```

and pass the full history on every call.

```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print(f"Mistral: {response.content}")
```

Flow:

```text
[SystemMessage, HumanMessage] → model → AIMessage
[SystemMessage, HumanMessage, AIMessage, HumanMessage] → model → AIMessage
```

Each turn appends both the user message and AI response to history.

---

## HuggingFace Chat Models

### Cloud Endpoint (HuggingFace Inference API)

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-R1-0528")
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is RAG?")
print(response.content)
```

Needs: `HUGGINGFACEHUB_ACCESS_TOKEN`

Uses HuggingFace's hosted inference API — no local GPU needed.

---

### Local Model (HuggingFacePipeline)

```python
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,
        "repetition_penalty": 1.03,
    },
)

chat_model = ChatHuggingFace(llm=hf)
response = chat_model.invoke("What is RAG?")
```

Runs model:

```text
Completely locally on your machine
```

No API key needed. Needs enough RAM/GPU.

---

## Streamlit Chatbot UI

Streamlit lets you build a web UI for your chatbot.

Key concepts:

### `st.session_state`

Persists data across reruns (like conversation history).

```python
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are helpful.")]
```

### `st.chat_input`

Input box at the bottom of the page.

### `st.chat_message`

Renders chat bubbles.

```python
with st.chat_message("user"):
    st.markdown(prompt)
```

### Mood Selection

```python
mood = st.selectbox("Select Mood", ["Helpful 😊", "Funny 😂", "Strict 🧑🏫"])
```

Changes the `SystemMessage` based on selected mood.

---

## Important Interview Points

- `.invoke()` takes a string or list of messages
- always use `response.content` to get the text
- conversation history must be manually maintained
- `SystemMessage` sets the AI's persona/behavior
- HuggingFace supports both cloud and local models
- `st.session_state` is how Streamlit persists state
- Groq = fast inference, Mistral = good balance of speed/quality
