import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pprint import pprint

load_dotenv()

async def ejecutar_arnes():
    # Nueva API: instanciacion directa sin gestor de contexto asincrono
    client = MultiServerMCPClient({
        "local_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["curso_oficial/modulo_2/mcp_server.py"],
        }
    })
        
    # Obtenemos las herramientas de forma asincrona directa
    tools = await client.get_tools()
    # Verificacion de herramientas cargadas
    print(f"Numero de herramientas cargadas: {len(tools)}")
    if len(tools) == 0:
        print("ERROR: No se cargaron herramientas del servidor MCP")
        print("Verifica que el servidor MCP esta correctamente implementado")
        return
    
    for i, tool in enumerate(tools, 1):
        print(f"Herramienta {i}: {tool.name} - {tool.description[:50]}...")
        
    # Obtencion de prompts. La nueva API devuelve un diccionario por servidor
    prompt_content = SystemMessage(content="" \
        "You are a helpful assistant with access to MCP tools. You MUST use the provided tools to answer questions."+
        "Available tools: """ + ", ".join([t.name for t in tools]) + 
        "When using a tool, you must provide the exact parameters as defined in the tool schema."+
        "Do not invent or hallucinate tools that are not in the list above.")        
        
    # Inicializamos el modelo. Se recomienda Llama 3.3 en Groq para un tool calling mas estable
    # Creamos el agente utilizando la funcion estandar de LangGraph
    model = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.0
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt_content
    ) # type: ignore

    config = {"configurable": {"thread_id": "1"}}

    # Invocacion asincrona del agente
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config # type: ignore
    )        
    pprint(response['messages'][-1].content) # type: ignore

if __name__ == "__main__":
    # Arrancamos el bucle de eventos principal
    asyncio.run(ejecutar_arnes())