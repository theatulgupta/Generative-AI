from dotenv import load_dotenv
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

load_dotenv()


# =========================
# OpenWeather Tool
# =========================

OW_KEY = os.getenv("OPENWEATHER_API_KEY")


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OW_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        return f"Could not get weather data for {city}. Error: {data.get('message')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]
    return f"The current temperature in {city} is {temp}°C with {description}."


# =========================
# Tavily News Tool
# =========================

TY_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TY_KEY)


@tool
def get_news(city: str) -> str:
    """Get the latest news for a given city."""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )
    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."

    news_list = []
    for result in results:
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("content", "")
        news_list.append(f"{title}\n{snippet}\nRead more: {url}")

    return "Latest news:\n\n" + "\n\n".join(news_list)


# =========================
# LLM Setup
# =========================


llm = ChatMistralAI(name="mistral-small-latest", temperature=0)

# Tool registry — maps tool name string to callable for manual execution
tools = {
    "get_weather": get_weather,
    "get_news": get_news,
}

# bind_tools tells the LLM which tools are available
llm_with_tools = llm.bind_tools([get_weather, get_news])


# =========================
# Manual Agent Loop
# =========================

messages = []

print("[bold green]City Intelligence System[/bold green]")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    # Inner loop — keeps running until LLM stops requesting tool calls
    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Human-in-the-loop: ask for approval before executing each tool
                confirm = input(
                    f"Agent wants to call '{tool_name}' with args {tool_args}. Approve? (yes/no): "
                )

                if confirm.lower() != "yes":
                    print("[red]Tool call denied.[/red]")
                    messages.append(ToolMessage(
                        content="Tool call denied by user.",
                        tool_call_id=tool_call["id"],
                    ))
                    continue

                # Execute the tool and send result back to LLM
                tool_result = tools[tool_name].invoke(tool_args)
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                ))

            continue  # LLM may call more tools after seeing results

        else:
            # No more tool calls — LLM has a final answer
            print(f"\n[bold cyan]Assistant:[/bold cyan] {result.content}\n")
            break
