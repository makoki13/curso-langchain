from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

load_dotenv()

#Paso 1: Conectar con el almacen vectorial existente
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
almacen_vectorial = Chroma(
    collection_name="manuals_empresa",
    embedding_function=modelo_embeddings,
    persist_directory="./chroma_db_local"
)

#Paso 2: Configurar el recuperador
#Usaremos una busqueda basica con 4 documentos para asegurar suficiente contexto.
recuperador = almacen_vectorial.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

#Paso 3: Funcion auxiliar para formatear el contexto
def formatear_documentos(docs):
    #Convierte la lista de objetos Document en una sola cadena de texto.
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

#Paso 4: Definir la plantilla de prompt RAG
prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto encargado de responder preguntas sobre los manuales y politicas de la empresa. "
            "Utiliza EXCLUSIVAMENTE la informacion proporcionada en el bloque de CONTEXTO para responder a la pregunta. "
            "Si la respuesta no se encuentra en el CONTEXTO, debes responder honestamente: 'No tengo informacion suficiente en los manuales para responder a esa pregunta'. "
            "No inventes informacion. Manten un tono profesional y conciso.\n\n"
            "CONTEXTO:\n{contexto}\n"
        ),
        (
            "human",
            "Pregunta: {pregunta}"
        )
    ]
)

#Paso 5: Inicializar el modelo y el parser
modelo_llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

parser_salida = StrOutputParser()

#Paso 6: Construir la cadena LCEL completa con RunnableParallel
#Aqui ocurre la magia del enrutamiento de datos.
cadena_rag = RunnableParallel(
    contexto=recuperador | formatear_documentos,
    pregunta=RunnablePassthrough()
) | prompt_rag | modelo_llm | parser_salida

#Paso 7: Ejecutar el pipeline
if __name__ == "__main__":
    print("--- PRUEBA 1: Pregunta dentro del contexto ---")
    pregunta_valida = "Cuantos dias de vacaciones tengo al ano y como solicito un reembolso?"
    respuesta_1 = cadena_rag.invoke(pregunta_valida)
    print(f"Pregunta: {pregunta_valida}")
    print(f"Respuesta: {respuesta_1}\n")

    print("--- PRUEBA 2: Pregunta fuera del contexto (Prueba anti-alucinacion) ---")
    pregunta_invalida = "Cual es la receta tradicional de la paella valenciana?"
    respuesta_2 = cadena_rag.invoke(pregunta_invalida)
    print(f"Pregunta: {pregunta_invalida}")
    print(f"Respuesta: {respuesta_2}\n")
