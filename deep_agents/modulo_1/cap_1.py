from deepagents import create_deep_agent

from utiles import modelos

agent = create_deep_agent(model=modelos.get_modelo())

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

print(result["messages"][-1].text)