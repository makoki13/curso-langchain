#pip install docling-slim[format-markdown]
#pip install langchain-docling

from dotenv import load_dotenv
from langchain_docling import DoclingLoader

load_dotenv()

'''
Paso 1: Carga de un archivo de texto local
Suponemos que tienes un archivo llamado politica_seguridad.txt en tu directorio.
Si no lo tienes, crea uno con un par de parrafos de texto.
'''
print("--- Cargando archivo de texto ---")

try:
    #loader_texto = TextLoader("politica_seguridad.txt", encoding="utf-8") # type: ignore
    loader = DoclingLoader(file_path="leccion_3\\politica_seguridad.txt")
    docs_texto = loader.load()
    
    for doc in docs_texto:
        print(f"Contenido (primeros 100 chars): {doc.page_content[:100]}...")
        print(f"Metadatos: {doc.metadata}")
except FileNotFoundError:
    print("Archivo de texto no encontrado. Continuando con los siguientes loaders...")

'''
Paso 2: Carga de un archivo CSV
Suponemos que tienes un archivo llamado empleados.csv con columnas: nombre, departamento, email.
'''
print("\n--- Cargando archivo CSV ---")
try:
    loader = DoclingLoader(file_path="leccion_3\\empleados.csv")
    docs_csv = loader.load()
    
    print(f"Total de documentos (filas) cargados: {len(docs_csv)}")
    if docs_csv:
        primer_doc_csv = docs_csv[0]
        print(f"Contenido de la primera fila: {primer_doc_csv.page_content}")
        print(f"Metadatos de la primera fila: {primer_doc_csv.metadata}")
except FileNotFoundError:
    print("Archivo CSV no encontrado. Continuando...")

'''
Paso 3: Carga de contenido web
'''
print("\n--- Cargando pagina web ---")
url_objetivo = "https://en.wikipedia.org/wiki/Large_language_model"
loader = DoclingLoader(file_path=url_objetivo)
docs_web = loader.load() # type: ignore

print(f"Total de documentos web cargados: {len(docs_web)}")
if docs_web:
    doc_web = docs_web[0]
    print(f"Contenido (primeros 150 chars): {doc_web.page_content[:150]}...")
    print(f"Metadatos web: {doc_web.metadata}")
