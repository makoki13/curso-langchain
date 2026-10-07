# pip install langchain-groq (siempre en entorno (.venv))
# pip show langchain-groq

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

nombre_modelo = "qwen/qwen3.8-27b"
modelo = ChatGroq(
    model=nombre_modelo,
    temperature=0.0
)
mensaje_usuario = HumanMessage(content="Explica en una sola frase que es un arnes de IA.")
mensajes = [mensaje_usuario]

respuesta = modelo.invoke(mensajes)

print("Respuesta del modelo:", respuesta.content)

if respuesta.usage_metadata:
    print("Tokens de entrada:", respuesta.usage_metadata['input_tokens'])
    print("Tokens de salida:", respuesta.usage_metadata['output_tokens'])
    print("Tokens totales:", respuesta.usage_metadata['total_tokens'])
else:
    print("Los metadatos de uso no están disponibles en esta versión")
    if hasattr(respuesta, 'response_metadata') and respuesta.response_metadata:
        print("Metadatos de respuesta:", respuesta.response_metadata)

print("\n--- Generando respuesta en tiempo real ---")
for chunk in modelo.stream(mensajes):
    print(chunk.content, end="", flush=True)
print()
