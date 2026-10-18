import asyncio
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    }
)

async def ejecutar_arnes():
    tools = await client.get_tools()

    model = ChatGroq(
                model="qwen/qwen3.8-27b",                
                temperature=0.0
            )
    agent = create_agent(
                model=model,
                tools=tools,
                checkpointer=InMemorySaver(),
                system_prompt="You are a travel agent. No follow up questions."
            ) # type: ignore

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on March 31st")]},
            config # type: ignore
    )

    pprint(response)
    print(response["messages"][-1].content)

if __name__ == "__main__":
    # Arrancamos el bucle de eventos principal
    asyncio.run(ejecutar_arnes())