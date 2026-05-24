from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528")

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)