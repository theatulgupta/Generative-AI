import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain.tools import tool


# @tool decorator turns a regular python function into a LangChain tool
# the docstring is what the LLM reads to decide when to use this tool — keep it clear
@tool
def get_greeting(name: str) -> str:
    """Generate a greeting message for a user"""
    return f"Hello {name}, Welcome to the GenAI World!"


result = get_greeting.invoke({"name": "Atul"})
print(result)

# tool exposes metadata the LLM uses internally
print(get_greeting.name)         # "get_greeting"
print(get_greeting.description)  # the docstring
print(get_greeting.args_schema)  # pydantic schema auto-generated from type hints
