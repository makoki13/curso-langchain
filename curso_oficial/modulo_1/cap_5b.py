from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import Dict, Any

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information. You MUST use this tool by providing a 'query' parameter containing the search terms."""
    return tavily_client.search(query)

system_prompt = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house.
Using the web_search tool, search the web for recipes that can be made with the ingredients they have.
In your first response, provide a numbered list of 3 to 5 recipe suggestions with a brief description of each. Do not provide full instructions yet.
If the user selects one of the options or asks you to develop a specific recipe, use the web_search tool again to find the full recipe instructions for that specific dish, and return the complete step-by-step instructions to the user.
"""

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0
)

memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "chef_session_01"}}

print("--- TURNO 1: Peticion inicial del usuario ---")
response_turno_1 = agent.invoke(
    {"messages": [HumanMessage(content="I have some leftover chicken and rice. What can I make?")]},
    config # type: ignore
)
print(response_turno_1['messages'][-1].content)
print("\n")

print("--- TURNO 2: El usuario pide desarrollar la segunda opcion ---")
response_turno_2 = agent.invoke(
    {"messages": [HumanMessage(content="I like the second option, the Chicken and Rice Casserole. Can you develop it and give me the full instructions?")]},
    config # type: ignore
)
print(response_turno_2['messages'][-1].content)