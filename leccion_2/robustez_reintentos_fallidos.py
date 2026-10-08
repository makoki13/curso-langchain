import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Simulamos un modelo que siempre falla con error de conexión
def modelo_que_siempre_falla(inputs):
    """Simula un modelo que siempre lanza un error de conexión."""
    print("  [SIMULACIÓN] Intentando conectar al modelo... FALLANDO")
    raise httpx.ConnectError("No se pudo conectar al servidor del modelo")

# Creamos un RunnableLambda que envuelve la función que falla
modelo_fallido = RunnableLambda(modelo_que_siempre_falla)

# Aplicamos la política de reintentos al modelo fallido
modelo_resiliente = modelo_fallido.with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(
        httpx.ConnectError,
        httpx.TimeoutException,
        ConnectionError,
        TimeoutError
    )
)

plantilla = ChatPromptTemplate.from_template("Resume en una frase: {texto}")
parser = StrOutputParser()

# La cadena incluye el modelo resiliente que siempre falla
cadena = plantilla | modelo_resiliente | parser

if __name__ == "__main__":
    print("Iniciando ejecución de la cadena con reintentos...\n")

    try:
        resultado = cadena.invoke({
            "texto": "La inteligencia artificial está transformando la industria."
        })
        print("Resultado exitoso:", resultado)
    except Exception as e:
        print("\n[EXCEPT] La cadena falló después de todos los reintentos")
        print(f"[EXCEPT] Tipo de error: {type(e).__name__}")
        print(f"[EXCEPT] Mensaje: {e}")
