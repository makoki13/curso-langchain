# pip install langchain-core langchain-openai langchain-chroma chromadb pypdf beautifulsoup4 ragas datasets numpy

import os
from typing import Any  # ← añadir al inicio del archivo

import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document
# Configuracion del sistema de logging para observabilidad
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
'''
============================================================
SECCION 1: CONFIGURACION GLOBAL DEL ARNES
============================================================
'''
class ConfiguracionArnes:
    # Clase que centraliza todos los parametros configurables del arnes.
    # En produccion, estos valores vendrian de variables de entorno.
    
    DIRECTORIO_DOCUMENTOS = "./leccion_3/ejercicio_1/documentos_empresa"
    DIRECTORIO_CHROMA = "./chroma_db_nexus"
    NOMBRE_COLECCION = "conocimiento_nexustech"
    MODELO_EMBEDDINGS = "text-embedding-3-small"
    MODELO_LLM = "gpt-4o-mini"
    TEMPERATURA = 0.0
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    NUM_DOCUMENTOS_RECUPERAR = 4
'''
============================================================
SECCION 2: FASE DE INDEXACION
============================================================
'''
def cargar_documentos(directorio: str) -> list:
    # Carga todos los archivos de texto del directorio corporativo.
    # Cada archivo se carga con metadatos que incluyen el nombre del archivo fuente.
    
    logger.info(f"Iniciando carga de documentos desde: {directorio}")
    
    loader = DirectoryLoader(
        path=directorio,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    
    documentos = loader.load()
    logger.info(f"Se cargaron {len(documentos)} documentos crudos.")
    return documentos


def enriquecer_metadatos(documentos: list) -> list:
    # Analiza el nombre del archivo fuente y anade un metadato de departamento.
    # Esto es critico para el filtrado posterior en las busquedas.
    
    documentos_enriquecidos = []
    
    for doc in documentos:
        fuente = doc.metadata.get("source", "")
        nombre_archivo = os.path.basename(fuente).lower()
        
        # Logica de clasificacion por departamento basada en el nombre del archivo.
        if "tecnico" in nombre_archivo or "infraestructura" in nombre_archivo:
            departamento = "tecnologia"
        elif "rrhh" in nombre_archivo or "recursos" in nombre_archivo:
            departamento = "recursos_humanos"
        elif "ventas" in nombre_archivo or "comercial" in nombre_archivo:
            departamento = "ventas"
        else:
            departamento = "general"
        
        # Creamos un nuevo documento con metadatos enriquecidos
        doc_enriquecido = Document(
            page_content=doc.page_content,
            metadata={
                "fuente": fuente,
                "departamento": departamento
            }
        )
        documentos_enriquecidos.append(doc_enriquecido)
        logger.info(f"Clasificado: {nombre_archivo} -> Departamento: {departamento}")
    
    return documentos_enriquecidos


def fragmentar_documentos(documentos: list) -> list:
    # Divide los documentos en chunks optimizados para la recuperacion semantica.
    
    logger.info(f"Fragmentando {len(documentos)} documentos con chunk_size={ConfiguracionArnes.CHUNK_SIZE}")
    
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=ConfiguracionArnes.CHUNK_SIZE,
        chunk_overlap=ConfiguracionArnes.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    fragmentos = divisor.split_documents(documentos)
    logger.info(f"Se generaron {len(fragmentos)} fragmentos.")
    return fragmentos


def construir_indice_vectorial(fragmentos: list):
    # Crea o actualiza el almacen vectorial con los fragmentos proporcionados.
    
    logger.info("Construyendo indice vectorial en ChromaDB...")
    
    #modelo_embeddings = OpenAIEmbeddings(model=ConfiguracionArnes.MODELO_EMBEDDINGS)
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    almacen = Chroma.from_documents(
        documents=fragmentos,
        embedding=modelo_embeddings,
        collection_name=ConfiguracionArnes.NOMBRE_COLECCION,
        persist_directory=ConfiguracionArnes.DIRECTORIO_CHROMA
    )
    
    logger.info("Indice vectorial construido y persistido exitosamente.")
    return almacen
'''
============================================================
SECCION 3: FASE DE CONSULTA Y GENERACION
============================================================
'''
def construir_cadena_rag(almacen_vectorial, filtro_departamento=None):
    '''
    Construye la cadena LCEL completa del asistente RAG.
    Si se proporciona un filtro de departamento, el retriever solo buscara
    en los documentos de ese departamento.
    '''
    # Modelo de lenguaje con reintentos para robustez
    """ modelo_llm = ChatOpenAI(
        model=ConfiguracionArnes.MODELO_LLM,
        temperature=ConfiguracionArnes.TEMPERATURA
    ).with_retry(stop_after_attempt=3) """

    modelo_llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

    
    # Configuracion del retriever con filtro opcional
    search_kwargs: dict[str, Any] = {"k": ConfiguracionArnes.NUM_DOCUMENTOS_RECUPERAR}
    if filtro_departamento:
        search_kwargs["filter"] = {"departamento": filtro_departamento}
        logger.info(f"Retriever configurado con filtro: departamento={filtro_departamento}")
    
    recuperador = almacen_vectorial.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # Funcion de formateo de documentos
    def formatear_contexto(docs):
        if not docs:
            return "No se encontraron documentos relevantes."
        return "\n\n---\n\n".join([
            f"[Fuente: {doc.metadata.get('fuente', 'desconocida')}] {doc.page_content}"
            for doc in docs
        ])
    
    # Plantilla de prompt con instrucciones anti-alucinacion estrictas
    prompt_rag = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres NexusBot, el asistente virtual de conocimiento corporativo de NexusTech. "
                "Tu mision es responder preguntas de los empleados basandote EXCLUSIVAMENTE en el "
                "CONTEXTO proporcionado. Reglas estrictas:\n"
                "1. Si la respuesta esta en el contexto, responde de forma clara, concisa y profesional.\n"
                "2. Si la respuesta NO esta en el contexto, responde exactamente: 'Lo siento, no he "
                "encontrado esa informacion en la documentacion disponible de NexusTech. Te recomiendo "
                "contactar al departamento correspondiente.'\n"
                "3. Cita la fuente del documento cuando sea posible.\n"
                "4. No inventes, asumas ni extrapoles informacion.\n\n"
                "CONTEXTO:\n{contexto}"
            ),
            (
                "human",
                "Pregunta del empleado: {pregunta}"
            )
        ]
    )
    
    # Parser de salida
    parser = StrOutputParser()
    
    # Funcion de observabilidad para registrar metricas
    def registrar_respuesta(respuesta: str) -> str:
        logger.info(f"Respuesta generada ({len(respuesta)} caracteres).")
        return respuesta
    
    # Construccion de la cadena LCEL completa con enrutamiento paralelo
    cadena = RunnableParallel(
        contexto=recuperador | formatear_contexto,
        pregunta=RunnablePassthrough()
    ) | prompt_rag | modelo_llm | parser | RunnableLambda(registrar_respuesta)
    
    return cadena
'''
============================================================
SECCION 4: INTERFAZ DE USUARIO Y EJECUCION
============================================================
'''
def ejecutar_asistente():
    # Funcion principal que orquesta todo el flujo del asistente.
    
    logger.info("=" * 60)
    logger.info("INICIANDO ASISTENTE CORPORATIVO NEXUSBOT")
    logger.info("=" * 60)
    
    # Verificar si el indice ya existe para evitar reindexar
    if os.path.exists(ConfiguracionArnes.DIRECTORIO_CHROMA):
        logger.info("Indice vectorial existente detectado. Cargando desde disco...")
        #modelo_embeddings = OpenAIEmbeddings(model=ConfiguracionArnes.MODELO_EMBEDDINGS)
        modelo_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        almacen = Chroma(
            collection_name=ConfiguracionArnes.NOMBRE_COLECCION,
            embedding_function=modelo_embeddings,
            persist_directory=ConfiguracionArnes.DIRECTORIO_CHROMA
        )
    else:
        logger.info("No se encontro indice previo. Ejecutando pipeline de indexacion completo...")
        documentos_crudos = cargar_documentos(ConfiguracionArnes.DIRECTORIO_DOCUMENTOS)
        documentos_enriquecidos = enriquecer_metadatos(documentos_crudos)
        fragmentos = fragmentar_documentos(documentos_enriquecidos)
        almacen = construir_indice_vectorial(fragmentos)
    
    # Bucle interactivo de preguntas
    print("\n" + "=" * 60)
    print("  NexusBot - Asistente de Conocimiento Corporativo")
    print("  Escribe 'salir' para terminar.")
    print("  Escribe 'filtro:tecnologia' para filtrar por departamento.")
    print("  Departamentos disponibles: tecnologia, recursos_humanos, ventas")
    print("=" * 60 + "\n")
    
    filtro_actual = None
    cadena_actual = construir_cadena_rag(almacen, filtro_actual)
    
    while True:
        entrada = input("\nTu pregunta: ").strip()
        
        if not entrada:
            continue
        
        if entrada.lower() == "salir":
            logger.info("Sesion terminada por el usuario.")
            break
        
        # Comando para cambiar el filtro de departamento
        if entrada.lower().startswith("filtro:"):
            nuevo_filtro = entrada.split(":")[1].strip().lower()
            if nuevo_filtro in ["tecnologia", "recursos_humanos", "ventas", "ninguno"]:
                filtro_actual = None if nuevo_filtro == "ninguno" else nuevo_filtro
                cadena_actual = construir_cadena_rag(almacen, filtro_actual)
                print(f"Filtro actualizado a: {filtro_actual or 'sin filtro (todos los departamentos)'}")
            else:
                print(f"Departamento no reconocido: {nuevo_filtro}")
            continue
        
        # Ejecutar la consulta RAG
        try:
            print("\nNexusBot esta pensando...")
            respuesta = cadena_actual.invoke(entrada)
            print(f"\nNexusBot: {respuesta}")
        except Exception as e:
            logger.error(f"Error critico en la cadena RAG: {e}")
            print("\nNexusBot: Lo siento, he experimentado un error tecnico. Por favor, intentalo de nuevo en unos momentos.")

if __name__ == "__main__":
    ejecutar_asistente()
