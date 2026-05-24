from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda,RunnablePassthrough

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()


code_prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a code generator"),
  ("user", "{topic}")
])


explain_prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant that explains code"),
  ("user", "Explain the following code in simple words: {topic}")
])

'''
seq = code_prompt | model | parser | explain_prompt | model | parser


result = seq.invoke({"topic" : "Palindrome in Python"})

print(result)
'''

# Now the problem is we are not getting the generated code from the chain, but our usecase needs both code and its explanation

# Solution - RunnablePassthrough
seq = code_prompt | model | parser
seq2 = RunnableParallel({
  "code" : RunnablePassthrough(),
  "explanation" : explain_prompt | model | parser
})

# chaining seq with seq2
chain = seq | seq2

result = chain.invoke({"topic" : "Palindrome in Python"})

print(result["code"])
print(result["explanation"])
