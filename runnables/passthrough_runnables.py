from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatMistralAI(name="mistral-small-2506")
parser = StrOutputParser()

# Prompt to generate code
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("user", "{topic}"),
])

# Prompt to explain code — receives the generated code as {topic}
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains code"),
    ("user", "Explain the following code in simple words: {topic}"),
])

# -------------------------------------------------------
# Problem: chaining code_prompt | model | parser | explain_prompt | model | parser
# loses the generated code — only the explanation comes out.
#
# Solution: RunnablePassthrough
# passes the generated code string through unchanged as "code"
# while the explanation branch processes it separately
# -------------------------------------------------------

# Step 1: generate code
seq = code_prompt | model | parser

# Step 2: split into code (passthrough) + explanation (new chain)
seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser,
})

# Full chain: topic → code → {code, explanation}
chain = seq | seq2

result = chain.invoke({"topic": "Palindrome in Python"})

print(result["code"])
print(result["explanation"])
