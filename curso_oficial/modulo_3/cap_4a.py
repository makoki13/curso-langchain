from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

from utiles import modelos
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.messages import AIMessage

load_dotenv()

large_model = modelos.get_modelo("openai")
standard_model = modelos.get_modelo()

@wrap_model_call
def state_based_model(request: ModelRequest, 
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)  

    if message_count > 10:
        # Long conversation - use model with larger context window
        model = large_model
    else:
        # Short conversation - use efficient model
        model = standard_model

    request = request.override(model=model)  

    return handler(request)

# Sólo una pregunta
agent = create_agent(
    model=standard_model,
    middleware=[state_based_model],
    system_prompt="You are roleplaying a real life helpful office intern."
)

response = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?")
        ]}
)

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])

# Varias preguntas
response = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?"),
        AIMessage(content="Yes, I gave it a light watering this morning."),
        HumanMessage(content="Has it grown much this week?"),
        AIMessage(content="It's sprouted two new leaves since Monday."),
        HumanMessage(content="Are the leaves still turning yellow on the edges?"),
        AIMessage(content="A little, but it's looking healthier overall."),
        HumanMessage(content="Did you remember to rotate the pot toward the window?"),
        AIMessage(content="I rotated it a quarter turn so it gets more even light."),
        HumanMessage(content="How often should we be fertilizing this plant?"),
        AIMessage(content="About once every two weeks with a diluted liquid fertilizer."),
        HumanMessage(content="When should we expect to have to replace the pot?")
        ]}
)

print(response["messages"][-1].response_metadata["model_name"])
print(response["messages"][-1].content)
