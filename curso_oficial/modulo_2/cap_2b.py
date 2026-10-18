from dotenv import load_dotenv
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


from utiles import modelos

load_dotenv()

class CustomState(AgentState):
    favourite_colour: str

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour, 
        "messages": [ToolMessage("Successfully updated favourite colour", tool_call_id=runtime.tool_call_id)]}
        )

@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"

agent = create_agent(
        model=modelos.get_modelo(""),
        tools=[update_favourite_colour, read_favourite_colour], 
        context_schema=CustomState,
        checkpointer=InMemorySaver(), 
        system_prompt="""You are a helpful assistant that can remember users' favourite colours.

When a user tells you their favourite colour, you MUST use the update_favourite_colour tool to save it.
When a user asks about their favourite colour, you MUST use the read_favourite_colour tool to retrieve it.

Always use the appropriate tools - do not just respond textually without using the tools."""
)
config = {"configurable": {"thread_id": "session_001"}}
    
response = agent.invoke(
    { "messages": [HumanMessage(content="My favourite colour is green")]},
    config=config # type: ignore
)
print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [HumanMessage(content="What's my favourite colour?")]}, # type: ignore
    config=config # type: ignore
)

print(response["messages"][-1].content)