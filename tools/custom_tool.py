import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain.tools import tool


# @tool turns a regular function into a LangChain tool
# the docstring becomes the tool's description — LLM reads it to decide when to use the tool
@tool
def get_greeting(name: str) -> str:
    """Generate a greeting message for a user"""
    return f"Hello {name}, Welcome to the GenAI World!"


result = get_greeting.invoke({"name": "Atul"})
print(result)

# Inspect tool metadata
print(get_greeting.name)         # function name
print(get_greeting.description)  # docstring
print(get_greeting.args_schema)  # auto-generated Pydantic schema from type hints
