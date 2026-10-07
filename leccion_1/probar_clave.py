import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("Clave API cargada correctamente en memoria.")
    print(f"La clave comienza con: {api_key[:5]}...")
else:
    print("Error: No se encontró la variable de entorno OPENAI_API_KEY.")
