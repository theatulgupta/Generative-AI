from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Components
model = ChatMistralAI(model="mistral-small-2506")

# Two different prompt
short_prompt = ChatPromptTemplate.from_template(
  "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
  "Explain {topic} in detail"
)

# Input
topic = "Runnables in langchain"

# Parser
parser = StrOutputParser()

'''
# Parallel Runnables
parallel_chain = RunnableParallel({
    "short" : short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
})

result = parallel_chain.invoke(topic)
short_result = result["short"]
detailed_result = result["detailed"]

print(short_result)
print(detailed_result)
'''

# With multiple input for each chain
'''
parallel_chain_2 = RunnableParallel({
    "short" : short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
})

result_2 = parallel_chain_2.invoke({"topic" : "Runnables in langchain", "audience" : "Beginners"})

short_result_2 = result_2["short"]
detailed_result_2 = result_2["detailed"]

print(short_result_2)
print(detailed_result_2)
'''

# With different inputs for each chain
'''
parallel_chain_3 = RunnableParallel({
    "short" : short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
})

result_3 = parallel_chain_3.invoke({
    "short" : {"topic" : "Runnables in langchain"},
    "detailed" : {"topic" : "Runnables in langchain"}
})

short_result_3 = result_3["short"]
detailed_result_3 = result_3["detailed"]

print(short_result_3)
print(detailed_result_3)
'''

# Using RunnableLambda
from langchain_core.runnables import RunnableLambda

lambda_chain = RunnableParallel({
  "short" : RunnableLambda(lambda x : x['short']) | short_prompt | model | parser,
  "detailed" : RunnableLambda(lambda x : x['detailed']) | detailed_prompt | model | parser
})

result_4 = lambda_chain.invoke({
  "short" : "Runnables in langchain",
  "detailed" : "Runnables in langchain"
})

print(result_4)
print(result_4["short"])
print(result_4["detailed"])
