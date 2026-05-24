import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from rich import print

load_dotenv()


# @tool turns a regular function into a LangChain tool
# the docstring is the tool description — LLM reads it to decide when to call this tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)


# Tool registry — maps tool name (string) to the actual callable
tools = {
    "get_text_length": get_text_length,
}

llm = ChatMistralAI(name="mistral-small-2506")

# bind_tools registers available tools with the LLM
# LLM will now include tool_calls in its response when it decides to use one
llm_with_tool = llm.bind_tools([get_text_length])

# -------------------------------------------------------
# Step 1: Send user query — LLM decides whether to call a tool
# -------------------------------------------------------
messages = []
prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

result = llm_with_tool.invoke(messages)
messages.append(result)

# -------------------------------------------------------
# Step 2: If LLM requested a tool call, execute it and append result
# -------------------------------------------------------
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_args = result.tool_calls[0]["args"]
    tool_result = tools[tool_name].invoke(tool_args)

    # ToolMessage links the result back to the specific tool call via tool_call_id
    messages.append(ToolMessage(
        content=str(tool_result),
        tool_call_id=result.tool_calls[0]["id"],
    ))

# -------------------------------------------------------
# Step 3: Send tool result back to LLM for the final answer
# -------------------------------------------------------
result = llm_with_tool.invoke(messages)
print(result.content)
