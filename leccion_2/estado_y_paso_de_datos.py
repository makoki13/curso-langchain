from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

load_dotenv()

modelo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)
parser = StrOutputParser()

# Paso 1: Definir los prompts y cadenas intermedias
prompt_clasificador = ChatPromptTemplate.from_template(
    "Clasifica la siguiente consulta del usuario en una de estas categorias exactas: 'tecnico', 'ventas', 'general'. "
    "Devuelve solo el nombre de la categoria, sin texto adicional.\n\n"
    "Consulta: {consulta}"
)

prompt_respuesta = ChatPromptTemplate.from_template(
    "Eres un asistente experto. La categoria de la consulta es {categoria}. "
    "Genera una respuesta profesional y detallada para la siguiente consulta del usuario:\n\n"
    "Consulta: {consulta}"
)

cadena_clasificacion = prompt_clasificador | modelo | parser

# Paso 2: Definir la funcion de logica personalizada
# Esta funcion tomara el estado enriquecido y construira el reporte final.
def construir_reporte_final(estado: dict) -> str:
    consulta_original = estado.get("consulta", "No disponible")
    categoria_detectada = estado.get("categoria", "desconocida")
    respuesta_generada = estado.get("respuesta", "Sin respuesta")

    reporte = (
        f"--- REPORTE DE ATENCION ---\n"
        f"Categoria: {categoria_detectada.upper()}\n"
        f"Consulta Original: {consulta_original}\n"
        f"Respuesta Generada: {respuesta_generada}\n"
        f"---------------------------"
    )
    return reporte

# Paso 3: Construir la cadena compleja con paso de estado
# Aqui es donde orquestamos todo.

cadena_completa = (
    # Primero, mantenemos la entrada original y anadimos la categoria.
    # RunnablePassthrough.assign ejecuta la cadena_clasificacion y guarda el resultado en la clave 'categoria'.
    # El diccionario de salida ahora tiene: {"consulta": "...", "categoria": "..."}
    RunnablePassthrough.assign(categoria=cadena_clasificacion)

    # Segundo, necesitamos generar la respuesta. Pero la respuesta depende tanto de la consulta como de la categoria.
    # Usamos RunnableParallel para ejecutar el prompt de respuesta pasando todo el diccionario actual,
    # y guardamos el resultado en la clave 'respuesta'.
    .assign(
        respuesta=RunnableParallel(
            consulta=RunnablePassthrough(),
            categoria=RunnablePassthrough()
        ) | prompt_respuesta | modelo | parser
    )

    # El diccionario de salida ahora tiene: {"consulta": "...", "categoria": "...", "respuesta": "..."}

    # Tercero, aplicamos nuestra funcion personalizada para formatear el reporte final.
    | RunnableLambda(construir_reporte_final)
)

# Paso 4: Ejecutar el arnes
if __name__ == "__main__":
    entrada_usuario = {"consulta": "Mi pantalla parpadea cada vez que abro el navegador, como lo soluciono?"}

    resultado_final = cadena_completa.invoke(entrada_usuario)
    print(resultado_final)
