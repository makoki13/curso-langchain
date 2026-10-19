from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.types import Command
#from pprint import pprint

from utiles import modelos


load_dotenv()

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    # take email from state
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    # fake email sending
    return "Email sent"

class EmailState(AgentState):
    email: str

agent = create_agent(
    model=modelos.get_modelo(),
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": [HumanMessage(content="Please read my email and send a response immediately. Send the reply now in the same thread.")],
        "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John."
    }, # type: ignore
    config=config # type: ignore
)

response = agent.invoke(
    Command(        
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # Edited action with tool name and args
                    "edited_action": {
                        # Tool name to call.
                        # Will usually be the same as the original action.
                        "name": "send_email",
                        # Arguments to pass to the tool.
                        "args": {"body": "This is the last straw, you're fired!"},
                    }
                }
            ]
        }
    ), 
    config=config # Same thread ID to resume the paused conversation # type: ignore
    )   

#pprint(response)
print(response["messages"][-1].content)