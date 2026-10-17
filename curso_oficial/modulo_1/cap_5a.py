from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from pprint import pprint


load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information. You MUST use this tool by providing a 'query' parameter containing the search terms."""
    return tavily_client.search(query)

system_prompt = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.
Using the web_search tool, search the web for recipes that can be made with the ingredients they have.
CRITICAL: When using the web_search tool, you must pass the search terms strictly as a JSON object with the key 'query'. Example: {"query": "chicken and rice recipes"}.
Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="I have some leftover chicken and rice. What can I make?")]},
    config # type: ignore
)

print(response['messages'][-1].content)
print("--- debug ---")
pprint(response)