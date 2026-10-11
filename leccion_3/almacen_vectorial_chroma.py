# pip install langchain-chroma chromadb

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Paso 1: Inicializar el modelo de incrustacion
# Utilizamos el mismo modelo que en el apartado anterior para mantener la coherencia en el espacio vectorial.
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Paso 2: Crear documentos de ejemplo con metadatos
# Imaginemos que estos son fragmentos extraidos de diferentes manuales de una empresa.
documentos = [
    Document(
        page_content="El procedimiento para reiniciar el servidor de produccion requiere aprobacion del director de infraestructura.",
        metadata={"fuente": "manual_infraestructura.pdf", "departamento": "tecnologia", "pagina": 12}
    ),
    Document(
        page_content="Los empleados tienen derecho a 22 dias laborables de vacaciones al ano, prorrateables por meses.",
        metadata={"fuente": "politicas_rrhh.pdf", "departamento": "recursos_humanos", "pagina": 5}
    ),
    Document(
        page_content="Para solicitar un reembolso de gastos de viaje, se debe utilizar el formulario F-104 y adjuntar facturas.",
        metadata={"fuente": "politicas_rrhh.pdf", "departamento": "recursos_humanos", "pagina": 18}
    ),
    Document(
        page_content="El cierre contable del trimestre debe completarse antes del dia 10 del mes siguiente.",
        metadata={"fuente": "manual_finanzas.pdf", "departamento": "finanzas", "pagina": 3}
    )
]

# Paso 3: Inicializar el almacen vectorial con Chroma
''' 
Utilizamos el metodo from_documents. Este metodo hace tres cosas automaticamente:
1. Extrae el texto de los documentos.
2. Llama al modelo de embeddings para vectorizar cada texto.
3. Inserta los vectores, los textos originales y los metadatos en la base de datos Chroma.
'''
print("Indexando documentos en ChromaDB...")
almacen_vectorial = Chroma.from_documents(
    documents=documentos,
    embedding=modelo_embeddings,
    collection_name="manuals_empresa",
    persist_directory="./chroma_db_local"
)
print("Indexacion completada.\n")

# Paso 4: Busqueda semantica basica
# Vamos a hacer una pregunta que semanticamente se parece a las vacaciones, pero sin usar las palabras exactas.
pregunta = "Cuantos dias de descanso anual me corresponden?"
print(f"Realizando busqueda basica para: '{pregunta}'")

resultados = almacen_vectorial.similarity_search(pregunta, k=2)
for i, doc in enumerate(resultados):
    print(f"Resultado {i+1}:")
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
    print("-" * 40)

# Paso 5: Busqueda semantica con filtrado de metadatos
# Ahora, el verdadero poder. Vamos a hacer la misma pregunta, pero vamos a obligar a la base de datos a buscar SOLO en el departamento de tecnologia.
print(f"\nRealizando busqueda filtrada para: '{pregunta}' (Filtro: departamento = tecnologia)")

resultados_filtrados = almacen_vectorial.similarity_search(
    query=pregunta,
    k=2,
    filter={"departamento": "tecnologia"}
)

if not resultados_filtrados:
    print("No se encontraron resultados en el departamento de tecnologia.")
else:
    for i, doc in enumerate(resultados_filtrados):
        print(f"Resultado filtrado {i+1}:")
        print(f"Contenido: {doc.page_content}")
        print(f"Metadatos: {doc.metadata}")
