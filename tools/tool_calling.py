import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from rich import print

load_dotenv()


# docstring is what the LLM reads to decide when to call this tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)


# registry so we can look up and call tools by name at runtime
tools = {
    "get_text_length": get_text_length,
}

llm = ChatMistralAI(name="mistral-small-2506")

# bind_tools tells the LLM which tools exist — it can now include tool_calls in its response
llm_with_tool = llm.bind_tools([get_text_length])

# step 1 — send the user query, LLM decides if it needs a tool
messages: list[BaseMessage] = []
prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

result = llm_with_tool.invoke(messages)
messages.append(result)

# step 2 — if LLM asked for a tool, run it and add the result back to messages
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_args = result.tool_calls[0]["args"]
    tool_result = tools[tool_name].invoke(tool_args)

    # tool_call_id links this result back to the specific tool call the LLM made
    messages.append(ToolMessage(
        content=str(tool_result),
        tool_call_id=result.tool_calls[0]["id"],
    ))

# step 3 — send everything back to LLM so it can give the final answer
result = llm_with_tool.invoke(messages)
print(result.content)
