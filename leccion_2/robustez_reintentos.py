from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Inicializamos el modelo.
modelo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

'''
Aplicamos la politica de reintentos directamente al modelo.
Le decimos que reintente un maximo de 3 veces.
'''
modelo_resiliente = modelo.with_retry(
    stop_after_attempt=3,
)

plantilla = ChatPromptTemplate.from_template("Resume en una frase: {texto}")
parser = StrOutputParser()

# La cadena ahora incluye el modelo resiliente.
cadena = plantilla | modelo_resiliente | parser

if __name__ == "__main__":
    try:
        resultado = cadena.invoke({"texto": "La inteligencia artificial esta transformando la industria."})
        print("Resultado exitoso:", resultado)
    except Exception as e:
        print("La cadena fallo despues de todos los reintentos:", e)
