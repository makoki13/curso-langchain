from deepagents import create_deep_agent
from utiles import modelos


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model=modelos.get_modelo(),
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Gandia, Valencia"}]}
)
print(result["messages"][-1].text)