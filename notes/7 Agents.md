# 7. Agents

An agent is:

```text
Agent = Brain (LLM) + Tools + Control Logic (our code)
```

The LLM acts as the brain — it:

- reasons about the problem
- decides which tool to call
- looks at the tool result
- decides next step
- repeats until it has a final answer

---

## Agent vs Simple Tool Calling

Simple tool calling:

```text
Fixed flow: User → LLM → Tool → LLM → Answer
```

Agent:

```text
Dynamic loop: User → LLM → Tool? → LLM → Tool? → LLM → Answer
```

Agent can call multiple tools, in any order, as many times as needed.

---

## ReAct Pattern

Most agents follow the ReAct pattern:

```text
Reason → Act → Observe → Reason → Act → ...
```

```text
Thought: I need to get the weather for Delhi
Action: call get_weather("Delhi")
Observation: "28°C with clear sky"
Thought: Now I have the weather, I can answer
Final Answer: The weather in Delhi is 28°C with clear sky.
```

---

## Manual Agent Loop

Build your own agent loop manually:

```python
llm_with_tools = llm.bind_tools([get_weather, get_news])
messages = []

while True:
    user_input = input("You: ")
    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_result = tools[tool_name].invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
            continue  # loop again — LLM may call more tools

        else:
            print(f"Assistant: {result.content}")
            break  # final answer reached
```

The inner `while True` loop keeps running until LLM stops calling tools.

---

## Human-in-the-Loop (Approval)

Add human approval before executing any tool:

```python
if result.tool_calls:
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        confirm = input(
            f"Agent wants to call '{tool_name}' with args {tool_args}. Approve? (yes/no): "
        )

        if confirm.lower() != "yes":
            messages.append(ToolMessage(
                content="Tool call denied by user.",
                tool_call_id=tool_call["id"]
            ))
            continue

        tool_result = tools[tool_name].invoke(tool_args)
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        ))
```

This gives you control over what the agent is allowed to do.

---

## Autonomous Agent with `create_agent`

LangChain provides `create_agent` to build agents without writing the loop manually:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant."
)

result = agent.invoke({
    "messages": [HumanMessage(content=user_input)]
})

final_message = result["messages"][-1]
print(final_message.content)
```

The agent handles the loop internally.

---

## `wrap_tool_call` Middleware

Intercepts every tool call before execution.

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def human_approval(request, handler):
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    approval = input(f"Approve {tool_name}({tool_args})? (yes/no): ")

    if approval.lower() != "yes":
        raise PermissionError(f"Tool '{tool_name}' execution denied.")

    return handler(request)

agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    middleware=[human_approval],
    system_prompt="You are a city assistant."
)
```

Cleaner way to add approval logic — separates concerns from the main loop.

---

## Tools Used in Agents Project

### get_weather (OpenWeather API)

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OW_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return f"The current temperature in {city} is {data['main']['temp']}°C with {data['weather'][0]['description']}."
```

Needs: `OPENWEATHER_API_KEY`

### get_news (Tavily)

```python
@tool
def get_news(city: str) -> str:
    """Get the latest news for a given city."""
    response = tavily_client.search(query=f"latest news in {city}", max_results=3)
    # format and return results
```

Needs: `TAVILY_API_KEY`

---

## Agent Message Flow

```text
User: "What's the weather and news in Mumbai?"

messages = [HumanMessage("What's the weather and news in Mumbai?")]

→ LLM: tool_calls = [get_weather("Mumbai"), get_news("Mumbai")]

→ ToolMessage(weather result)
→ ToolMessage(news result)

→ LLM: final answer combining both results
```

---

## Files Summary

- `agents.py` - Manual agent loop with human approval per tool call
- `agents_autonomous.py` - Autonomous agent using `create_agent`
- `agents_wrap_tool_call.py` - Agent with `wrap_tool_call` middleware for approval

---

## Important Interview Points

- agent = LLM + tools + control loop
- LLM decides which tool to call and with what args
- agent loop runs until LLM stops calling tools
- human-in-the-loop adds approval before tool execution
- `create_agent` abstracts the loop — good for production
- `wrap_tool_call` middleware is cleaner than inline approval logic
- `ToolMessage` must have matching `tool_call_id`
- agents can call multiple tools in sequence or parallel
- ReAct = Reason + Act pattern used by most agents
