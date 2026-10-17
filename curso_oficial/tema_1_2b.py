from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from pprint import pprint
from langchain_groq import ChatGroq  # noqa: E402

@tool("raiz_cuadrada", description="Calcula la raiz cuadrada de un número")
def raiz_cuadrada(x: float) -> float:
    return x ** 0.5

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(model=model, tools=[raiz_cuadrada])

pregunta = HumanMessage(content="What is the square root of 467?")
promptSistema = SystemMessage(content="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number.")

response = agent.invoke(
    {"messages": [promptSistema,pregunta,]} # El SystemMessage debe ser siempre el primer mensaje de la lista.
)

print ("-- response['messages'][-1].content")
print(response['messages'][-1].content)

print ("-- response['messages']")
pprint(response['messages'])

print("-- Busqueda segura de tool_calls")
for mensaje in response['messages']:
    if mensaje.type == 'ai' and hasattr(mensaje, 'tool_calls') and mensaje.tool_calls:
        print(mensaje.tool_calls)