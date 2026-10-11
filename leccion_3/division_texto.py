from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#Texto de ejemplo extenso para demostrar la division.
texto_largo = (
    "Los arneses de IA son fundamentales para la ingenieria de software moderna. "
    "Permiten orquestar modelos de lenguaje de forma robusta y escalable. "
    "Sin embargo, construir un arnes no es solo conectar APIs. "
    "Requiere un diseno cuidadoso de los flujos de datos, la gestion de estado y la tolerancia a fallos. "
    "La recuperacion aumentada por generacion es un patron clave en estos sistemas. "
    "Permite a los modelos acceder a conocimiento privado y actualizado. "
    "Para que el RAG funcione bien, la division del texto en fragmentos es critica. "
    "Un mal chunking puede arruinar la calidad de la recuperacion semantica."
)

'''
Configuracion del divisor recursivo.
chunk_size: Tamano maximo de cada fragmento en caracteres.
chunk_overlap: Cuantos caracteres se solapan entre fragmentos consecutivos.
'''
divisor = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

#Ejecutamos la division.
fragmentos = divisor.split_text(texto_largo)

#Inspeccionamos los resultados.
print(f"Texto original: {len(texto_largo)} caracteres.")
print(f"Total de fragmentos generados: {len(fragmentos)}\n")

for i, fragmento in enumerate(fragmentos):
    print(f"--- Fragmento {i+1} ---")
    print(fragmento)
    print(f"Longitud: {len(fragmento)} caracteres.")
    
    #Si no es el ultimo fragmento, mostramos el solapamiento con el siguiente.
    if i < len(fragmentos) - 1:
        siguiente_fragmento = fragmentos[i+1]
        solapamiento_real = siguiente_fragmento[:30]
        print(f"[SOLAPAMIENTO CON EL SIGUIENTE]: '{solapamiento_real}'")
    print("-" * 50)
