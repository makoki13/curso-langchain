from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# Modelo principal
modelo_principal = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Modelo de respaldo
modelo_respaldo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

'''
Aplicamos la politica de reintentos al modelo principal primero.
Si despues de reintentar sigue fallando, pasara al respaldo.
'''
modelo_principal_resiliente = modelo_principal.with_retry(stop_after_attempt=2)

# Configuramos la cadena de fallback.
modelo_con_respaldo = modelo_principal_resiliente.with_fallbacks([modelo_respaldo])

plantilla = ChatPromptTemplate.from_template("Cual es la capital de {pais}?")
parser = StrOutputParser()

cadena_resiliente = plantilla | modelo_con_respaldo | parser

if __name__ == "__main__":
    try:
        resultado = cadena_resiliente.invoke({"pais": "Francia"})
        print("Respuesta:", resultado)
    except Exception as e:
        print("Incluso los modelos de respaldo fallaron:", e)
