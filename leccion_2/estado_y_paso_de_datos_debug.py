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

modelo = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0)
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

# Paso 3: Construir la cadena en dos etapas para poder ver el estado intermedio

# Etapa 1: Clasificacion y enriquecimiento inicial
paso_clasificacion = RunnablePassthrough.assign(categoria=cadena_clasificacion)

# Etapa 2: Generacion de respuesta y reporte final
paso_respuesta_y_reporte = (
    RunnablePassthrough.assign(
        respuesta=RunnableParallel(
            consulta=RunnablePassthrough(),
            categoria=RunnablePassthrough()
        ) | prompt_respuesta | modelo | parser
    )
    | RunnableLambda(construir_reporte_final)
)

# Paso 4: Ejecutar el arnes en dos fases
if __name__ == "__main__":
    entrada_usuario = {"consulta": "Mi pantalla parpadea cada vez que abro el navegador, como lo soluciono?"}

    print("=" * 60)
    print("FASE 1: CLASIFICACION")
    print("=" * 60)
    
    # Ejecutamos solo el primer paso
    resultado_intermedio = paso_clasificacion.invoke(entrada_usuario)
    
    print("\nEstado despues del primer assign:")
    print(resultado_intermedio)
    print(f"\nTipo de 'categoria': {type(resultado_intermedio['categoria'])}")
    print(f"Valor de 'categoria': '{resultado_intermedio['categoria']}'")

    """ print("\n" + "=" * 60)
    print("FASE 2: GENERACION DE RESPUESTA Y REPORTE")
    print("=" * 60)
    
    # Continuamos con el segundo paso usando el resultado intermedio
    resultado_final = paso_respuesta_y_reporte.invoke(resultado_intermedio)
    
    print("\nResultado final:")
    print(resultado_final) """