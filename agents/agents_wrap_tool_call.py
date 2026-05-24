from dotenv import load_dotenv
import os
import requests

from rich import print
from tavily import TavilyClient

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()


# =========================
# OpenWeather Tool
# =========================

OW_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OW_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found.")


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OW_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        return f"Weather fetch failed: {data.get('message')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"{city}: {temp}°C with {desc}"


# =========================
# Tavily News Tool
# =========================

TY_KEY = os.getenv("TAVILY_API_KEY")
if not TY_KEY:
    raise ValueError("TAVILY_API_KEY not found.")

tavily_client = TavilyClient(api_key=TY_KEY)


@tool
def get_news(city: str) -> str:
    """Get latest news for a city."""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )
    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    output = []
    for r in results:
        title = r.get("title", "No title")
        content = r.get("content", "")
        url = r.get("url", "")
        output.append(f"{title}\n{content}\n{url}")

    return "\n\n".join(output)


# =========================
# Human Approval Middleware
# =========================

# @wrap_tool_call intercepts every tool call before execution
# cleaner than inline approval logic inside the agent loop
@wrap_tool_call
def human_approval(request, handler):
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})

    print(f"\n[yellow]Tool Request:[/yellow] {tool_name}({tool_args})")

    approval = input("Approve tool execution? (yes/no): ")

    if approval.lower() != "yes":
        raise PermissionError(f"Tool '{tool_name}' execution denied.")

    # Call the original handler to actually execute the tool
    return handler(request)


# =========================
# LLM + Agent Setup
# =========================


llm = ChatMistralAI(name="mistral-small-latest")

agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    middleware=[human_approval],   # middleware runs before every tool execution
    system_prompt="You are a city assistant.",
)


# =========================
# Chat Loop
# =========================

print("[bold green]City Intelligence System[/bold green]")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        final_message = result["messages"][-1]
        print(f"\n[bold cyan]Assistant:[/bold cyan]\n{final_message.content}\n")

    except PermissionError as e:
        print(f"\n[bold red]Permission Denied:[/bold red]\n{e}\n")

    except Exception as e:
        print(f"\n[bold red]Error:[/bold red]\n{str(e)}\n")
