from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from utiles import modelos


load_dotenv()

tavily_client = TavilyClient()

db = SQLDatabase.from_uri("sqlite:///curso_oficial/modulo_2/Chinook.db")


@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

@tool
def sql_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query) # type: ignore
    except Exception as e:
        return f"Error: {e}"

@dataclass
class UserRole:
    user_role: str = "external"

@wrap_model_call
def dynamic_tool_call(request: ModelRequest, 
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

    """Dynamically call tools based on the runtime context"""

    user_role = request.runtime.context.user_role # type: ignore
    
    if user_role == "internal":
        pass # internal users get access to all tools
    else:
        tools = [web_search] # external users only get access to web search
        request = request.override(tools=tools)  # type: ignore

    return handler(request)

agent = create_agent(
    model=modelos.get_modelo(),
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call], # type: ignore
    context_schema=UserRole
) # type: ignore

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    #context={"user_role": "external"}
    context={"user_role": "internal"}
)

print(response["messages"][-1].content)