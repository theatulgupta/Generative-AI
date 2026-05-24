# 4. Runnables & LCEL

LCEL = LangChain Expression Language

A way to compose LangChain components using:

```text
| (pipe operator)
```

Just like Unix pipes.

---

## Why LCEL?

Without LCEL (manual):

```python
formatted_prompt = prompt.format_messages(topic="Runnables")
response = model.invoke(formatted_prompt)
parsed = parser.parse(response)
```

With LCEL:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "Runnables"})
```

Benefits:
- cleaner code
- composable
- supports parallel execution
- lazy evaluation
- streaming support

---

## What is a Runnable?

Any LangChain component that implements:

```text
.invoke()
.stream()
.batch()
```

All of these are Runnables:
- `ChatPromptTemplate`
- `ChatMistralAI`
- `StrOutputParser`
- `RunnableParallel`
- `RunnablePassthrough`
- `RunnableLambda`

---

## 1. Sequence Runnables (Chain)

Components connected one after another.

```text
Input → Prompt → Model → Parser → Output
```

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "Agentic AI"})
print(result)
```

Data flows left to right:
1. `prompt` formats the input into messages
2. `model` generates a response
3. `parser` extracts the string

---

## 2. Parallel Runnables

Run multiple chains simultaneously on the same input.

```text
         ┌── short_chain ──┐
Input ───┤                 ├── {"short": ..., "detailed": ...}
         └── detail_chain ─┘
```

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "short": short_prompt | model | parser,
    "detailed": detailed_prompt | model | parser
})

result = parallel_chain.invoke({"topic": "Runnables in LangChain"})
print(result["short"])
print(result["detailed"])
```

Output is a dict with keys matching the parallel chain keys.

---

### RunnableLambda with Parallel

When each branch needs different input:

```python
from langchain_core.runnables import RunnableLambda

lambda_chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser
})

result = lambda_chain.invoke({
    "short": "Runnables in LangChain",
    "detailed": "Runnables in LangChain"
})
```

`RunnableLambda` wraps any Python function as a Runnable.

---

## 3. Passthrough Runnables

Problem:

```text
In a chain, intermediate values are lost
```

Example:

```python
seq = code_prompt | model | parser | explain_prompt | model | parser
```

Here the generated code is lost — only explanation comes out.

Solution:

```text
RunnablePassthrough
```

Passes the input through unchanged alongside the chain output.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

seq = code_prompt | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),       # passes generated code as-is
    "explanation": explain_prompt | model | parser
})

chain = seq | seq2

result = chain.invoke({"topic": "Palindrome in Python"})
print(result["code"])
print(result["explanation"])
```

Flow:

```text
topic → code_prompt | model | parser → generated_code (string)
                                            ↓
                              ┌─── "code": passthrough ───────────────┐
                              └─── "explanation": explain | model | parser ─┘
                                            ↓
                              {"code": "...", "explanation": "..."}
```

---

## Runnable Summary

| Runnable | Purpose |
|---|---|
| `prompt \| model \| parser` | Sequential chain |
| `RunnableParallel` | Run multiple chains at once |
| `RunnablePassthrough` | Pass input unchanged to output |
| `RunnableLambda` | Wrap any Python function as Runnable |

---

## `.invoke()` vs `.batch()` vs `.stream()`

```python
chain.invoke({"topic": "AI"})          # single input
chain.batch([{"topic": "AI"}, ...])    # list of inputs
chain.stream({"topic": "AI"})          # streaming output
```

---

## Important Interview Points

- LCEL uses `|` pipe operator to compose chains
- every LangChain component is a Runnable
- `RunnableParallel` runs branches simultaneously (faster)
- `RunnablePassthrough` solves the "lost intermediate value" problem
- `RunnableLambda` wraps any Python function into a chain
- chains are lazy — nothing runs until `.invoke()` is called
- output of one Runnable becomes input of the next
