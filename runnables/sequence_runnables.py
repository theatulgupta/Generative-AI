from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")
model = ChatMistralAI(name="mistral-small-2506")
parser = StrOutputParser()

# manual way — same thing as the chain below, just step by step
# formatted_prompt = prompt.format_messages(topic="Runnables in LangChain")
# response = model.invoke(formatted_prompt)
# result = parser.invoke(response)
# print(result)

# LCEL chain — pipe operator passes output of each step as input to the next
# prompt formats the input → model generates response → parser extracts plain string
chain = prompt | model | parser
result = chain.invoke({"topic": "Agentic AI"})
print(result)
