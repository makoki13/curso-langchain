from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from utiles import modelos
from langchain.messages import HumanMessage, AIMessage
#from pprint import pprint

load_dotenv()

agent = create_agent(
    model=modelos.get_modelo(""),
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=modelos.get_modelo(""),
            trigger=("tokens", 100),
            keep=("messages", 1)
        )
    ],
)
#agent = modelos.get_agente("",tools=tools,prompt_sistema=system_prompt)

response = agent.invoke(
    {"messages": [
        HumanMessage(content="What is the capital of the moon?"),
        AIMessage(content="The capital of the moon is Lunapolis."),
        HumanMessage(content="What is the weather in Lunapolis?"),
        AIMessage(content="Skies are clear, with a high of 120C and a low of -100C."),
        HumanMessage(content="How many cheese miners live in Lunapolis?"),
        AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
        HumanMessage(content="Do you think the cheese miners' union will strike?"),
        AIMessage(content="Yes, because they are unhappy with the new president."),
        HumanMessage(content="If you were Lunapolis' new president how would you respond to the cheese miners' union?"),
        ]},
    {"configurable": {"thread_id": "1"}}
)
#pprint(response)
print(response["messages"][0].content)

