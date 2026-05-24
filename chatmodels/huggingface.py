from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# HuggingFaceEndpoint — uses HuggingFace's hosted Inference API (no local GPU needed)
# Needs: HUGGINGFACEHUB_ACCESS_TOKEN in .env
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528")

# Wrap with ChatHuggingFace to get the standard chat model interface
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)
