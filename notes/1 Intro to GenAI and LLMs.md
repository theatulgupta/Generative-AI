# 1. Introduction to Generative AI & LLMs

Generative AI is a type of AI that can:
- generate
- create
- produce

new content like text, images, audio, code.

In simple words:

```text
Input (Prompt) → AI Model → Generated Output
```

---

## What is an LLM?

LLM = Large Language Model

A neural network trained on:

```text
Massive amounts of text data
```

to understand and generate human language.

Examples:
- GPT-4 (OpenAI)
- Gemini (Google)
- Mistral
- LLaMA (Meta)
- DeepSeek

---

## How LLMs Work (High Level)

```text
Text → Tokens → Embeddings → Transformer → Output Tokens → Text
```

Steps:
1. Input text split into tokens
2. Tokens converted to vectors (embeddings)
3. Transformer processes with attention mechanism
4. Output tokens generated one by one (auto-regressive)
5. Tokens decoded back to text

---

## What LLMs Are NOT Good At

```text
LLMs have limitations
```

Problems:
- no real-time data (knowledge cutoff)
- not reliable for calculations
- cannot call APIs on their own
- cannot access your private data
- hallucinations (confidently making things up)
- no memory by default across sessions

Example:

```text
"What is today's weather in Delhi?"
```

LLM cannot answer this accurately without tools.

---

## Key Terms

### Token

Smallest unit of text an LLM processes.

```text
"I love NLP"
→ ["I", " love", " NLP"]
```

Roughly:

```text
1 token ≈ 4 characters ≈ 0.75 words
```

---

### Context Window

Maximum tokens an LLM can process at once.

```text
Input tokens + Output tokens ≤ Context Window
```

Examples:
- GPT-4: 128k tokens
- Gemini 1.5 Pro: 1M tokens
- Mistral Small: 32k tokens

---

### Temperature

Controls randomness of output.

```text
temperature = 0    → deterministic, focused
temperature = 0.9  → creative, varied
temperature > 1    → very random / chaotic
```

Use:
- `temperature=0` for factual/coding tasks
- `temperature=0.9` for creative writing

---

### Max Tokens

Limits the length of the generated response.

```python
model = ChatMistralAI(model="mistral-small-2506", temperature=0.9, max_tokens=20)
```

---

## Types of LLM APIs

### Proprietary (Paid / Cloud)
- OpenAI (GPT-4)
- Google (Gemini)
- Mistral AI
- Groq (fast inference)

### Open Source (Free / Self-hosted)
- Meta LLaMA
- DeepSeek
- TinyLlama
- Mistral (open weights)

---

## Important Interview Points

- LLMs predict next token based on probability
- temperature controls creativity vs determinism
- context window = memory limit per conversation
- LLMs don't have real-time data by default
- tools and agents extend LLM capabilities
- hallucination is a major challenge in production
- open source models can run locally on your machine
