from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# HuggingFaceEndpoint connects to HuggingFace's hosted inference API
# so the model runs on their servers, not your machine
# needs HUGGINGFACEHUB_API_TOKEN in .env
llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-R1-0528")

# ChatHuggingFace wraps the endpoint so we can use .invoke() like any other chat model
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)
