#online mcp_server

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from pprint import pprint
from langchain_core.messages import HumanMessage

client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "run",
                "python",
                "-m",
                "mcp_server_time",
                "--local-timezone=America/New_York"
            ]
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
        ) # type: ignore

    question = HumanMessage(content="What time is it?")

    response = await agent.ainvoke(
        {"messages": [question]}
    )
    pprint(response)

if __name__ == "__main__":
    # Arrancamos el bucle de eventos principal
    asyncio.run(ejecutar_arnes())