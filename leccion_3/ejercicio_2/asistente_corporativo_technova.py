from typing import Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# Configuramos el sistema de logs para observar el comportamiento del arnes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("AsistenteTechNova")

load_dotenv()

def generar_base_conocimiento():
    '''
    Genera una lista de objetos Document con metadatos enriquecidos
    que simulan la documentacion corporativa de TechNova Solutions.
    '''
    documentos = [
        # Documento 1: Manual de Infraestructura
        Document(
            page_content=(
                "Procedimiento de reinicio del servidor de produccion PRD-01. "
                "Antes de iniciar el reinicio, el ingeniero de guardia debe verificar que no haya "
                "transacciones activas en la base de datos principal. Para ello, ejecutar el comando "
                "check_active_transactions.sh en la terminal del nodo maestro. "
                "Si hay transacciones activas, esperar a su finalizacion o contactar al equipo de "
                "desarrollo para forzar el cierre. Una vez confirmada la inactividad, ejecutar "
                "sudo systemctl restart tecnova-api. El tiempo estimado de reinicio es de 45 segundos. "
                "Durante este periodo, el balanceador de carga redirigira el trafico al nodo PRD-02."
            ),
            metadata={
                "fuente": "manual_infraestructura_v3.pdf",
                "departamento": "tecnologia",
                "autor": "Carlos Mendez",
                "nivel_confidencialidad": "interno",
                "pagina": 12
            }
        ),
        
        # Documento 2: Politica de Vacaciones
        Document(
            page_content=(
                "Politica de vacaciones y permisos de TechNova Solutions. "
                "Todos los empleados a tiempo completo tienen derecho a 22 dias laborables de "
                "vacaciones remuneradas por ano natural. Los dias se prorratean proporcionalmente "
                "en funcion de la fecha de incorporacion. Las solicitudes de vacaciones deben "
                "presentarse con un minimo de 15 dias naturales de antelacion a traves del portal "
                "de empleado en el modulo de Recursos Humanos. El responsable directo tiene un plazo "
                "de 5 dias habiles para aprobar o denegar la solicitud. En caso de denegacion, "
                "se debe proporcionar una justificacion por escrito."
            ),
            metadata={
                "fuente": "politicas_rrhh_2024.pdf",
                "departamento": "recursos_humanos",
                "autor": "Laura Ramirez",
                "nivel_confidencialidad": "publico",
                "pagina": 5
            }
        ),
        
        # Documento 3: Procedimiento de Reembolsos
        Document(
            page_content=(
                "Procedimiento para la solicitud de reembolso de gastos de viaje corporativo. "
                "El empleado debe completar el formulario F-104 disponible en la intranet y "
                "adjuntar las facturas originales o copias digitalizadas legibles. "
                "Los gastos elegibles incluyen transporte, alojamiento y manutencion con un "
                "limite diario de 60 euros para comidas. Los gastos de alcohol no son reembolsables. "
                "El formulario debe entregarse al departamento de Finanzas dentro de los 30 dias "
                "naturales siguientes a la finalizacion del viaje. El plazo de pago del reembolso "
                "es de 15 dias habiles desde la aprobacion."
            ),
            metadata={
                "fuente": "politicas_rrhh_2024.pdf",
                "departamento": "recursos_humanos",
                "autor": "Laura Ramirez",
                "nivel_confidencialidad": "publico",
                "pagina": 18
            }
        ),
        
        # Documento 4: Cierre Contable
        Document(
            page_content=(
                "Procedimiento de cierre contable trimestral. El equipo de contabilidad debe "
                "completar la conciliacion bancaria antes del dia 5 del mes siguiente al cierre "
                "del trimestre. Los informes de ingresos y gastos deben ser validados por el "
                "director financiero antes del dia 8. La presentacion final al consejo de "
                "administracion debe realizarse antes del dia 10 del mes siguiente. "
                "Cualquier discrepancia superior a 5000 euros debe ser reportada inmediatamente "
                "al comite de auditoria mediante el formulario de incidencias financieras F-201."
            ),
            metadata={
                "fuente": "manual_finanzas_Q4.pdf",
                "departamento": "finanzas",
                "autor": "Pedro Sanchez",
                "nivel_confidencialidad": "confidencial",
                "pagina": 3
            }
        ),
        
        # Documento 5: Onboarding de nuevos empleados
        Document(
            page_content=(
                "Guia de onboarding para nuevos empleados de TechNova Solutions. "
                "En el primer dia, el nuevo empleado debe presentarse en recepcion a las 9:00 "
                "con su documento de identidad y numero de seguridad social. El equipo de "
                "Recursos Humanos le entregara su equipo informatico, que incluye un portatil "
                "corporativo con acceso VPN preconfigurado, un monitor externo y un kit de "
                "perifericos. Durante la primera semana, el empleado debera completar el curso "
                "obligatorio de ciberseguridad en la plataforma de formacion interna. "
                "El periodo de prueba tiene una duracion de 3 meses para contratos indefinidos "
                "y de 1 mes para contratos temporales."
            ),
            metadata={
                "fuente": "guia_onboarding_2024.pdf",
                "departamento": "recursos_humanos",
                "autor": "Laura Ramirez",
                "nivel_confidencialidad": "interno",
                "pagina": 1
            }
        ),
        
        # Documento 6: Politica de Ciberseguridad
        Document(
            page_content=(
                "Politica de ciberseguridad de TechNova Solutions. Todos los empleados deben "
                "utilizar autenticacion de doble factor para acceder a los sistemas corporativos. "
                "Las contrasenas deben tener un minimo de 12 caracteres, incluyendo mayusculas, "
                "minusculas, numeros y caracteres especiales. El cambio de contrasena es obligatorio "
                "cada 90 dias. Queda estrictamente prohibido el uso de dispositivos USB personales "
                "en los equipos corporativos. Cualquier incidente de seguridad debe reportarse "
                "inmediatamente al equipo de tecnologia a traves del canal de Slack #seguridad "
                "o mediante el telefono de guardia 900-123-456."
            ),
            metadata={
                "fuente": "manual_infraestructura_v3.pdf",
                "departamento": "tecnologia",
                "autor": "Carlos Mendez",
                "nivel_confidencialidad": "confidencial",
                "pagina": 25
            }
        )
    ]
    
    return documentos

def indexar_documentos(documentos, persist_dir="./chroma_technova"):
    # Ingiere, fragmenta e indexa los documentos en la base de datos vectorial.
    
    logger.info(f"Iniciando indexacion de {len(documentos)} documentos...")
    
    # Configurar el divisor de texto
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Dividir los documentos en fragmentos
    fragmentos = divisor.split_documents(documentos)
    logger.info(f"Se generaron {len(fragmentos)} fragmentos tras el chunking.")
    
    # Inicializar el modelo de incrustacion
    #modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Crear el almacen vectorial
    almacen = Chroma.from_documents(
        documents=fragmentos,
        embedding=modelo_embeddings,
        collection_name="tecnova_corporativo",
        persist_directory=persist_dir
    )
    
    logger.info("Indexacion completada exitosamente.")
    return almacen

def construir_retriever(almacen, departamento=None, k=4):
    # Construye un retriever con filtro opcional de departamento.
        
    search_kwargs: dict[str, Any] = {"k": k}
    
    # Si se especifica un departamento, anadimos el filtro de metadatos
    if departamento:
        search_kwargs["filter"] = {"departamento": departamento}
        logger.info(f"Retriever configurado con filtro: departamento={departamento}")
    else:
        logger.info("Retriever configurado sin filtros (busqueda global)")
    
    return almacen.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

def construir_cadena_rag(almacen, departamento=None):
    # Construye el pipeline RAG completo con observabilidad y manejo de errores.
    
    #modelo_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    modelo_llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

    #modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
    # Configurar el retriever
    recuperador = construir_retriever(almacen, departamento=departamento, k=4)
    
    # Funcion de formateo de documentos con citas de fuente
    def formatear_con_fuentes(docs):
        if not docs:
            return "No se encontraron documentos relevantes."
        partes = []
        for i, doc in enumerate(docs, 1):
            fuente = doc.metadata.get("fuente", "desconocida")
            pagina = doc.metadata.get("pagina", "N/A")
            partes.append(
                f"[Documento {i} - Fuente: {fuente}, Pagina: {pagina}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(partes)
    
    # Funcion de observabilidad
    def registrar_consulta(resultado):
        logger.info(f"Respuesta generada. Longitud: {len(resultado)} caracteres.")
        return resultado
    
    # Definir la plantilla de prompt anti-alucinacion
    prompt_rag = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres el asistente de conocimiento corporativo de TechNova Solutions. "
                "Tu funcion es responder preguntas de los empleados basandote EXCLUSIVAMENTE "
                "en el CONTEXTO proporcionado a continuacion. "
                "Reglas estrictas:\n"
                "1. Si la respuesta esta en el contexto, proporciona una respuesta clara y "
                "detallada, citando las fuentes entre corchetes.\n"
                "2. Si la respuesta NO esta en el contexto, responde exactamente: "
                "'No dispongo de informacion sobre ese tema en la documentacion de TechNova. "
                "Te recomiendo contactar al departamento correspondiente.'\n"
                "3. Nunca inventes informacion, procedimientos o cifras que no aparezcan "
                "explicitamente en el contexto.\n"
                "4. Si la pregunta es ambigua, pide aclaracion al usuario.\n\n"
                "CONTEXTO:\n{contexto}\n"
            ),
            (
                "human",
                "Pregunta del empleado: {pregunta}"
            )
        ]
    )
    
    # Parser de salida
    parser = StrOutputParser()
    
    # Ensamblar la cadena LCEL
    cadena = (
        RunnableParallel(
            contexto=recuperador | formatear_con_fuentes,
            pregunta=RunnablePassthrough()
        )
        | prompt_rag
        | modelo_llm
        | parser
        | RunnableLambda(registrar_consulta)
    )
    
    return cadena

def ejecutar_asistente():
    # Funcion principal que orquesta todo el flujo del asistente.
    
    print("=" * 70)
    print("   ASISTENTE DE CONOCIMIENTO CORPORATIVO - TECHNOVA SOLUTIONS")
    print("=" * 70)
    print()
    
    # Generar e indexar la base de conocimiento
    documentos = generar_base_conocimiento()
    almacen = indexar_documentos(documentos)
    
    # Definir las preguntas de prueba con sus configuraciones
    pruebas = [
        {
            "pregunta": "Como reinicio el servidor de produccion?",
            "departamento": "tecnologia",
            "descripcion": "Pregunta tecnica con filtro de departamento"
        },
        {
            "pregunta": "Cuantos dias de vacaciones me corresponden y como los solicito?",
            "departamento": "recursos_humanos",
            "descripcion": "Pregunta de RRHH con filtro de departamento"
        },
        {
            "pregunta": "Que necesito para solicitar un reembolso de viaje y cual es el limite de comidas?",
            "departamento": None,
            "descripcion": "Pregunta global sin filtro (busca en todos los departamentos)"
        },
        {
            "pregunta": "Cual es la receta de la paella valenciana?",
            "departamento": None,
            "descripcion": "Pregunta fuera de contexto (prueba anti-alucinacion)"
        },
        {
            "pregunta": "Que debo hacer en mi primer dia de trabajo y cuanto dura el periodo de prueba?",
            "departamento": "recursos_humanos",
            "descripcion": "Pregunta de sintesis (requiere combinar info de varios documentos)"
        }
    ]
    
    # Ejecutar cada prueba
    for i, prueba in enumerate(pruebas, 1):
        print(f"\n{'=' * 70}")
        print(f"PRUEBA {i}: {prueba['descripcion']}")
        print(f"{'=' * 70}")
        print(f"Pregunta: {prueba['pregunta']}")
        print(f"Departamento: {prueba['departamento'] or 'Todos'}")
        print("-" * 70)
        
        try:
            # Construir la cadena para esta configuracion especifica
            cadena = construir_cadena_rag(
                almacen, 
                departamento=prueba["departamento"]
            )
            
            # Ejecutar la consulta
            respuesta = cadena.invoke(prueba["pregunta"])
            
            print(f"\nRespuesta del asistente:\n{respuesta}")
            
        except Exception as e:
            logger.error(f"Error en la prueba {i}: {str(e)}")
            print("\n[ERROR] El asistente no pudo procesar esta consulta.")
            print(f"Detalle tecnico: {str(e)}")
            print("Por favor, contacte al equipo de tecnologia.")
        
        print()

if __name__ == "__main__":
    ejecutar_asistente()
