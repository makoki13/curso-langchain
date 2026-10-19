from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from pprint import pprint
from utiles import modelos
from typing import Callable

import warnings
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*", category=UserWarning)

load_dotenv()

@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"

class AuthenticatedState(AgentState):
    authenticated: bool

@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"

@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password: # type: ignore
        return Command(update={
            "authenticated": True, 
            "messages": [ToolMessage(
                "Successfully authenticated", 
                tool_call_id=runtime.tool_call_id)]
        })
    else:
        return Command(update={
            "authenticated": False,
            "messages": [ToolMessage(
                "Authentication failed", 
                tool_call_id=runtime.tool_call_id)]
        })

# Middleware para herramientas dinámicas - CORREGIDO
class DynamicToolMiddleware(AgentMiddleware):
    """Allow read inbox and send email tools only if user provides correct email and password"""
    
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        # Este método intercepta la llamada al MODELO, no la ejecución de herramientas
        authenticated = request.state.get("authenticated")
        
        if authenticated:
            tools = [check_inbox, send_email]
        else:
            tools = [authenticate]

        # Modificar las herramientas disponibles para el modelo
        request = request.override(tools=tools) # type: ignore
        return handler(request)

# Middleware para prompt dinámico
class DynamicPromptMiddleware(AgentMiddleware):
    """Generate system prompt based on authentication status"""
    
    def get_system_prompt(self, request: ModelRequest) -> str:
        authenticated = request.state.get("authenticated")
        
        if authenticated:
            return """You are a helpful assistant that can check the inbox and send emails. 
Your first step after authentication is to check the inbox."""
        else:
            return "You are a helpful assistant that can authenticate users."

agent = create_agent(
    modelos.get_modelo(),
    tools=[authenticate, check_inbox, send_email],
    checkpointer=InMemorySaver(),
    state_schema=AuthenticatedState,
    context_schema=EmailContext,
    middleware=[
        DynamicToolMiddleware(), 
        DynamicPromptMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "authenticate": False,
                "check_inbox": False,
                "send_email": True,
            })
    ] # type: ignore
) # type: ignore

config = {"configurable": {"thread_id": "1"}}

# Bucle para continuar hasta que haya interrupción
response = agent.invoke(
    {"messages": [HumanMessage(content="julie@example.com, password123")]},
    context=EmailContext(),
    config=config
)

print("Paso 1:", response['messages'][-1].content)

# Continuar hasta que haya interrupción
max_iterations = 5
iteration = 0

while '__interrupt__' not in response and iteration < max_iterations:
    iteration += 1
    print(f"\n=== Iteración {iteration} ===")
    
    # Dar instrucciones al modelo para que intente enviar el email
    user_input = "Yes, send the reply now"
    
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    print(f"Respuesta: {response['messages'][-1].content}")
    
    if '__interrupt__' in response:
        print("\n=== INTERRUPCIÓN DETECTADA ===")
        break

if '__interrupt__' in response:
    interrupt_data = response['__interrupt__'][0].value
    pprint(interrupt_data)
    
    if 'action_requests' in interrupt_data:
        email_args = interrupt_data['action_requests'][0]['args']
        print(f"\nTo: {email_args.get('to')}")
        print(f"Subject: {email_args.get('subject')}")
        print(f"Body: {email_args.get('body')}")
    
    # Aprobar el envío
    response = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
    
    print("\n=== Después de aprobar ===")
    print(response['messages'][-1].text)
else:
    print("\nNo se alcanzó la interrupción después de múltiples iteraciones")