from typing import cast

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_groq import ChatGroq

load_dotenv()

# Cargar plantilla few-shot
plantilla_raw = load_prompt("plantilla_extractor_entidades.json")
plantilla_few_shot = cast(ChatPromptTemplate, plantilla_raw)

# Formatear con el texto del usuario
mensajes = plantilla_few_shot.format_messages(
    texto_usuario="La cámara Sony A7 IV tiene un sensor de 33 megapíxeles y cuesta 2500 dólares."
)

# Ejecutar con el modelo
modelo = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
respuesta = modelo.invoke(mensajes)

print("Respuesta:", respuesta.content)
