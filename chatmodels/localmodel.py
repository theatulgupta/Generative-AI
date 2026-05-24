from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100, 
        "do_sample": False,
        "repetition_penalty": 1.03,
    },
) 

chat_model = ChatHuggingFace(llm=hf)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)