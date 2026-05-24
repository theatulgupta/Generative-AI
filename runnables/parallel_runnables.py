from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = ChatMistralAI(name="mistral-small-2506")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

# basic parallel — both branches get the same input dict, run at the same time
# parallel_chain = RunnableParallel({
#     "short": short_prompt | model | parser,
#     "detailed": detailed_prompt | model | parser,
# })
# result = parallel_chain.invoke({"topic": "Runnables in LangChain"})
# print(result["short"])
# print(result["detailed"])

# RunnableLambda — when each branch needs to pull a different key from the input
# lambda extracts just the value that branch needs before passing to its prompt
lambda_chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser,
})

result = lambda_chain.invoke({
    "short": "Runnables in LangChain",
    "detailed": "Runnables in LangChain",
})

print(result["short"])
print(result["detailed"])
