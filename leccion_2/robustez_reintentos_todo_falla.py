import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Función que simula un modelo que siempre falla
def modelo_que_siempre_falla(inputs):
    print("  [SIMULACIÓN] Modelo principal FALLANDO")
    raise httpx.ConnectError("No se pudo conectar al servidor")

def modelo_respaldo_que_tambien_falla(inputs):
    print("  [SIMULACIÓN] Modelo de respaldo TAMBIÉN FALLANDO")
    raise httpx.ConnectError("No se pudo conectar al servidor de respaldo")

# Modelos simulados que siempre fallan

modelo_principal = RunnableLambda(modelo_que_siempre_falla)
modelo_respaldo = RunnableLambda(modelo_respaldo_que_tambien_falla)

# Aplicar reintentos al modelo principal
modelo_principal_resiliente = modelo_principal.with_retry(
    stop_after_attempt=2,
    retry_if_exception_type=(httpx.ConnectError,)
)

# Configurar fallback al modelo de respaldo
modelo_con_respaldo = modelo_principal_resiliente.with_fallbacks([modelo_respaldo])

plantilla = ChatPromptTemplate.from_template("Cual es la capital de {pais}?")
parser = StrOutputParser()

cadena_resiliente = plantilla | modelo_con_respaldo | parser

# Función de último recurso
def manejar_error_fallido(entrada):
    print("  [SIMULACIÓN] Ejecutando fallback de mensaje amigable")
    return "Lo sentimos, nuestro sistema de inteligencia artificial está experimentando problemas técnicos. Por favor, inténtalo más tarde."

# Añadir fallback final
cadena_con_mensaje_error = cadena_resiliente.with_fallbacks(
    [RunnableLambda(manejar_error_fallido)]
)

if __name__ == "__main__":
    print("Iniciando ejecución con todos los niveles de fallback...\n")

    try:
        resultado = cadena_con_mensaje_error.invoke({"pais": "Francia"})
        print(f"\nRespuesta final: {resultado}")
    except Exception as e:
        print(f"\nIncluso el fallback final falló: {e}")
