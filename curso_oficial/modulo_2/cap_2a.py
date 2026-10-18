from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime

load_dotenv()

@dataclass
class ColourContext:
    favourite_colour: str = "red"
    least_favourite_colour: str = "yellow"

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour

@tool
def get_least_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

model = ChatGroq(
    model="qwen/qwen3.8-27b",                
    temperature=0.0
            )
agent = create_agent(
    model=model,
    tools=[get_favourite_colour, get_least_favourite_colour],
    context_schema=ColourContext  
) # type: ignore

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
    #context=ColourContext(favourite_colour="pink")
)
print(response["messages"][-1].content)

'''
Otra forma es definir un contexto propio:

contexto_usuario = ColourContext(
    favourite_colour="blue",
    least_favourite_colour="green"
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=contexto_usuario
)
'''