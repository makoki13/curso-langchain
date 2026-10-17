from langgraph.checkpoint.memory import InMemorySaver  
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq  # noqa: E402
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(model=model, checkpointer=InMemorySaver())


question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [question]},
    config,   # type: ignore
)
pprint(response)

question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,   # type: ignore
)

question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,   # type: ignore
)

print("\n")
pprint(f"{response['messages'][-1].content}") # type: ignore