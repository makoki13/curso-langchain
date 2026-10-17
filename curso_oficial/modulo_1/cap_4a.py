from PIL import Image
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq  # noqa: E402
from langchain.agents import create_agent

import base64

ruta_imagen = "curso_oficial\\modulo_1\\moon.png"
imagen_cargada = Image.open(ruta_imagen)

with open(ruta_imagen, "rb") as fichero_imagen:
    datos_binarios = fichero_imagen.read()
    datos_base64 = base64.b64encode(datos_binarios).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this capital"},
    {"type": "image", "base64": datos_base64, "mime_type": "image/png"}
])

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(model=model)


response = agent.invoke(
    {"messages": [multimodal_question]}
)

print(response['messages'][-1].content)