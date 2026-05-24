from dotenv import load_dotenv
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from rich import print

load_dotenv()


# =========================
# OpenWeather Tool
# =========================

OW_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OW_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found in environment variables.")


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
if not TY_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

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
# LLM + Agent Setup
# =========================

# Use model= (not name=) — name= silently falls back to default 'mistral-small'
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

# create_agent abstracts the tool-calling loop — no manual while loop needed
agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant.",
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

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Last message in the result is the final AI response
    final_message = result["messages"][-1]
    print(f"\n[bold cyan]Assistant:[/bold cyan] {final_message.content}\n")
