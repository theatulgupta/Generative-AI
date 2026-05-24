from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatMistralAI(name="mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("user", "{topic}"),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains code"),
    ("user", "Explain the following code in simple words: {topic}"),
])

# problem: if we chain code_prompt | model | parser | explain_prompt | model | parser
# the generated code gets lost — only the explanation comes out at the end
# we need both the code AND the explanation

# solution: RunnablePassthrough
# after seq generates the code string, we split into two branches:
# one branch just passes the code through unchanged
# other branch takes the code and generates an explanation
seq = code_prompt | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser,
})

chain = seq | seq2

result = chain.invoke({"topic": "Palindrome in Python"})

print(result["code"])
print(result["explanation"])
