from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two prompts asking for different levels of detail on the same topic
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

# -------------------------------------------------------
# Basic RunnableParallel — both branches get the same input dict
# -------------------------------------------------------
# parallel_chain = RunnableParallel({
#     "short": short_prompt | model | parser,
#     "detailed": detailed_prompt | model | parser,
# })
# result = parallel_chain.invoke({"topic": "Runnables in LangChain"})
# print(result["short"])
# print(result["detailed"])

# -------------------------------------------------------
# RunnableLambda — extract the right key before passing to each branch
# useful when each branch needs a different slice of the input
# -------------------------------------------------------
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
