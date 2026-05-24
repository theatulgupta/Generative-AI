from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# TavilySearch — real-time web search tool designed for LLM use
# max_results limits how many search results are returned
search_tool = TavilySearch(max_results=3)

model = ChatMistralAI(name="mistral-small-2506")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.
    Summarize the following news into clear bullet points: {news} (in 50 words only)
    """
)

# Chain: prompt → LLM → plain string
chain = prompt | model | parser

# Fetch real-time news via Tavily, then summarize with the LLM
news_result = search_tool.invoke({"query": "Latest AI news of 2026"})
result = chain.invoke({"news": news_result})

print(result)
