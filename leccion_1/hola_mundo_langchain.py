from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)

mensaje_usuario = HumanMessage(content="Explica en una sola frase que es un arnes de IA.")
mensajes = [mensaje_usuario]

# respuesta es un objeto de la clase AIMessage (https://reference.langchain.com/python/langchain-core/messages/ai/AIMessage)
respuesta = modelo.invoke(mensajes)

print("Respuesta del modelo:", respuesta.content)

if respuesta.usage_metadata:
    print("Tokens de entrada:", respuesta.usage_metadata['input_tokens'])
    print("Tokens de salida:", respuesta.usage_metadata['output_tokens'])
    print("Tokens totales:", respuesta.usage_metadata['total_tokens'])
else:
    print("Los metadatos de uso no están disponibles en esta versión")
    # Alternativa: usar response_metadata si está disponible
    if hasattr(respuesta, 'response_metadata') and respuesta.response_metadata:
        print("Metadatos de respuesta:", respuesta.response_metadata)
