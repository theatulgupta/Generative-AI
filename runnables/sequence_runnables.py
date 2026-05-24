from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
  "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

# Step by Step - Manual Flow

'''
# Step 1: Format the prompt
formatted_prompt = prompt.format_messages(topic = "Runnables in langchain")

# Step 2: Get the response from the model
response = model.invoke(formatted_prompt)

# Step 3: Parse the response
parsed_response = parser.parse(response)

print(parsed_response.content)
'''


# Step by Step - Sequence Runnables
chain = prompt | model | parser
result = chain.invoke("Agentic AI")
print(result)
