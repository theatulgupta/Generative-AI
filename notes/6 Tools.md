# 6. Tools in LangChain

LLMs are limited:

```text
- no real-time data
- can't do math reliably
- can't call APIs
- can't access your data
```

Tools solve this:

```text
LLM decides WHAT to do
Tools actually DO the task
Our code controls HOW everything works together
```

---

## What is a Tool?

A tool is a:

```text
Python function that an LLM can call
```

The LLM doesn't run the function itself — it tells your code to run it, then gets the result back.

```text
User Query → LLM → "I need to call tool X with args Y"
                         ↓
                   Your code runs tool X
                         ↓
                   Result sent back to LLM
                         ↓
                   LLM gives final answer
```

---

## Creating a Custom Tool

Use the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def get_greeting(name):
    """Generate a greeting message for a user"""
    return f"Hello {name}, Welcome to the GenAI World!"
```

The docstring is critical:

```text
Docstring = tool description
```

The LLM reads the docstring to decide when to use the tool.

---

## Tool Properties

```python
print(get_greeting.name)          # "get_greeting"
print(get_greeting.description)   # "Generate a greeting message for a user"
print(get_greeting.args_schema)   # Pydantic schema of arguments
```

---

## Invoking a Tool Directly

```python
result = get_greeting.invoke({"name": "Atul"})
print(result)  # "Hello Atul, Welcome to the GenAI World!"
```

---

## Tool Calling Flow (Manual)

### Step 1: Bind tools to LLM

```python
llm_with_tool = llm.bind_tools([get_text_length])
```

This tells the LLM what tools are available.

### Step 2: LLM decides to call a tool

```python
messages = [HumanMessage("How many characters in 'Hello World'?")]
result = llm_with_tool.invoke(messages)
```

If LLM wants to use a tool, `result.tool_calls` is populated:

```python
result.tool_calls
# [{"name": "get_text_length", "args": {"text": "Hello World"}, "id": "..."}]
```

### Step 3: Execute the tool

```python
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_args = result.tool_calls[0]["args"]
    tool_result = tools[tool_name].invoke(tool_args)
```

### Step 4: Send result back to LLM

```python
from langchain_core.messages import ToolMessage

tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=result.tool_calls[0]["id"]
)
messages.append(tool_message)

final_result = llm_with_tool.invoke(messages)
print(final_result.content)
```

Full message flow:

```text
[HumanMessage] 
→ LLM → [AIMessage with tool_calls]
→ [ToolMessage with result]
→ LLM → [AIMessage with final answer]
```

---

## Complete Tool Calling Example

```python
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

tools = {"get_text_length": get_text_length}

llm = ChatMistralAI(name="mistral-small-2506")
llm_with_tool = llm.bind_tools([get_text_length])

messages = [HumanMessage(input("You: "))]
result = llm_with_tool.invoke(messages)
messages.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_result = tools[tool_name].invoke(result.tool_calls[0]["args"])
    messages.append(ToolMessage(
        content=str(tool_result),
        tool_call_id=result.tool_calls[0]["id"]
    ))

final = llm_with_tool.invoke(messages)
print(final.content)
```

---

## Search Tool (Tavily)

Tavily is a search API designed for LLMs.

```python
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)
news_result = search_tool.invoke({"query": "Latest AI news of 2026"})
```

Returns real-time web search results.

Needs: `TAVILY_API_KEY`

### Using Search Tool in a Chain

```python
chain = prompt | llm | parser

news_result = search_tool.invoke({"query": "Latest AI news of 2026"})
result = chain.invoke({"news": news_result})
print(result)
```

---

## ToolMessage

Special message type to return tool results back to the LLM.

```python
from langchain_core.messages import ToolMessage

ToolMessage(
    content="11",                    # result of the tool
    tool_call_id=result.tool_calls[0]["id"]  # must match the tool call ID
)
```

The `tool_call_id` links the result to the specific tool call the LLM made.

---

## Important Interview Points

- `@tool` decorator turns any function into a LangChain tool
- docstring is the tool description — LLM uses it to decide when to call
- `bind_tools()` registers tools with the LLM
- `result.tool_calls` is populated when LLM wants to use a tool
- `ToolMessage` sends tool results back to the LLM
- `tool_call_id` must match between tool call and tool message
- tools extend LLM with real-time data, calculations, API access
- Tavily is a search tool built specifically for LLM use cases
