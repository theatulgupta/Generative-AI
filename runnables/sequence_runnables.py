from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Prompt Template — {topic} is the dynamic variable
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser — extracts plain string from AIMessage
parser = StrOutputParser()

# -------------------------------------------------------
# Manual flow (step by step — same as the chain below)
# -------------------------------------------------------
# formatted_prompt = prompt.format_messages(topic="Runnables in LangChain")
# response = model.invoke(formatted_prompt)
# result = parser.invoke(response)
# print(result)

# -------------------------------------------------------
# LCEL chain — pipe operator connects components left to right
# prompt → model → parser
# -------------------------------------------------------
chain = prompt | model | parser
result = chain.invoke({"topic": "Agentic AI"})
print(result)
