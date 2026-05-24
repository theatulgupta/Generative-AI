from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

# HuggingFacePipeline — downloads and runs the model locally (no API key needed)
# Requires enough RAM; TinyLlama-1.1B is small enough to run on CPU
hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,        # greedy decoding — deterministic output
        "repetition_penalty": 1.03, # slight penalty to avoid repetitive text
    },
)

# Wrap with ChatHuggingFace to use the standard .invoke() interface
chat_model = ChatHuggingFace(llm=hf)

response = chat_model.invoke("What is Retrieval Augmented Generation? Tell in short.")
print(response.content)
