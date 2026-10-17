from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq  # noqa: E402

load_dotenv()

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(model=model)

question = HumanMessage(content="Hola, mi nombre es Pablo y mi color favorito es el rojo")

response = agent.invoke(
    {"messages": [question]} 
)

print(response['messages'][-1].content)

question = HumanMessage(content="¿Cual es mi color favorito?")

response = agent.invoke(
    {"messages": [question]} 
)

print(response['messages'][-1].content)