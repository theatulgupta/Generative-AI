from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

# HuggingFacePipeline downloads and runs the model locally on your machine
# no API key needed, but needs enough RAM
# TinyLlama is only 1.1B params so it runs fine on CPU
hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,         # False = greedy, always picks the most likely token
        "repetition_penalty": 1.03, # slightly penalizes repeating the same words
    },
)

# wrap with ChatHuggingFace so we get the same .invoke() interface as other models
chat_model = ChatHuggingFace(llm=hf)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)
