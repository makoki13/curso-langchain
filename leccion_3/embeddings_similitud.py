# ==============================================================================
# IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ==============================================================================
#pip install sentence-transformers
#pip install langchain-huggingface

# load_dotenv carga las variables de entorno desde el archivo .env
# Aunque HuggingFace no requiere API key para modelos locales,
# es buena práctica mantener esta línea para futuras integraciones
from dotenv import load_dotenv

# numpy es una librería fundamental para computación numérica en Python
# La usaremos para calcular la similitud coseno entre vectores
import numpy as np

# HuggingFaceEmbeddings es el wrapper de LangChain para modelos de embeddings
# de Hugging Face. Permite usar cientos de modelos gratuitos que se ejecutan
# localmente en tu máquina, sin necesidad de API keys ni costes.
from langchain_huggingface import HuggingFaceEmbeddings

# Cargamos las variables de entorno del archivo .env
load_dotenv()


# ==============================================================================
# PASO 1: INICIALIZAR EL MODELO DE EMBEDDINGS
# ==============================================================================

'''
Paso 1: Inicializar el modelo de embeddings (incrustaciones)

Utilizamos el modelo de embeddings de Hugging Face.
paraphrase-multilingual-MiniLM-L12-v2 es un modelo multilingüe gratuito
que funciona bien tanto en español como en inglés.

Características del modelo:
- Dimensiones: 384 (cada texto se convierte en un vector de 384 números)
- Multilingüe: Funciona en más de 50 idiomas incluyendo español
- Ligero: ~471 MB, se ejecuta rápidamente en CPU
- Gratuito: No requiere API key ni pagos
- Local: Se descarga la primera vez y se cachea en tu máquina
'''
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ==============================================================================
# PASO 2: DEFINIR UN CONJUNTO DE FRASES PARA COMPARAR
# ==============================================================================

# Definimos un conjunto de frases en español para demostrar
# cómo los embeddings capturan la similitud semántica.
# Nota cómo las frases 1-2 son similares (perro/jardín),
# las frases 3-4 son similares (bolsa/mercado),
# y la frase 5 es diferente (comer manzanas).
frases = [
    "El perro esta corriendo en el parque",
    "Un canino juega feliz en el jardin",
    "El mercado de valores experimento una fuerte caida",
    "Las acciones de tecnologia cayeron en la bolsa hoy",
    "Me gusta comer manzanas rojas"
]


# ==============================================================================
# PASO 3: GENERAR LOS VECTORES DE EMBEDDINGS
# ==============================================================================

'''
Paso 3: Generar los vectores de embeddings (incrustaciones)

El método embed_documents convierte una lista de textos en una lista de vectores.
Cada vector es una representación numérica del significado semántico del texto.

Proceso interno:
1. El texto se tokeniza (se divide en subpalabras)
2. Los tokens pasan por el modelo transformer
3. Se obtiene un vector de 384 dimensiones que representa el significado

Textos con significado similar tendrán vectores cercanos en el espacio vectorial.
'''
print("Generando vectores de embeddings...")
vectores = modelo_embeddings.embed_documents(frases)

# Información sobre los vectores generados
print(f"Número de frases: {len(frases)}")
print(f"Dimensión de cada vector: {len(vectores[0])}")
print(f"Primeros 5 valores del primer vector: {vectores[0][:5]}\n")

# Los vectores son arrays de números flotantes entre -1 y 1 aproximadamente
# Cada posición en el vector captura una dimensión diferente del significado


# ==============================================================================
# PASO 4: DEFINIR LA FUNCIÓN DE SIMILITUD COSENO
# ==============================================================================

'''
Paso 4: Definir la función de Similitud Coseno

La similitud coseno mide el ángulo entre dos vectores en el espacio vectorial.
- Valor 1.0: Vectores idénticos (mismo significado)
- Valor 0.0: Vectores ortogonales (significados no relacionados)
- Valor -1.0: Vectores opuestos (significados contrarios)

Fórmula matemática:
cos(θ) = (A · B) / (||A|| × ||B||)

Donde:
- A · B es el producto punto de los vectores
- ||A|| y ||B|| son las normas (longitudes) de los vectores

Esta métrica es ideal para embeddings porque:
- Es insensible a la magnitud de los vectores
- Se enfoca puramente en la dirección (significado)
- Es computacionalmente eficiente
'''
def similitud_coseno(vector_a, vector_b):
    """
    Calcula el coseno del ángulo entre dos vectores.
    
    Parámetros:
    - vector_a: Lista o array de números (primer vector)
    - vector_b: Lista o array de números (segundo vector)
    
    Retorna:
    - float: Valor entre -1 y 1 indicando la similitud
    """
    # Convertimos las listas a arrays de numpy para operaciones vectoriales
    a = np.array(vector_a)
    b = np.array(vector_b)
    
    # Producto punto: suma de los productos de elementos correspondientes
    # Captura cuánto "apuntan en la misma dirección" los vectores
    producto_punto = np.dot(a, b)
    
    # Norma euclidiana: longitud del vector (raíz cuadrada de la suma de cuadrados)
    # Normaliza para que la magnitud no afecte la similitud
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    
    # Similitud coseno: producto punto dividido por el producto de las normas
    # El resultado está en el rango [-1, 1]
    return producto_punto / (norma_a * norma_b)


# ==============================================================================
# PASO 5: COMPARAR LA SIMILITUD ENTRE PARES DE FRASES
# ==============================================================================

'''
Paso 5: Comparar la similitud entre pares de frases

Vamos a comparar la primera frase con todas las demás para demostrar
cómo el modelo captura similitudes semánticas incluso cuando las palabras
son diferentes (perro vs canino, parque vs jardín).

Resultados esperados:
- Frase 2 (canino/jardín): Alta similitud con frase 1 (perro/parque)
- Frases 3-4 (bolsa/mercado): Baja similitud con frase 1
- Frase 5 (manzanas): Muy baja similitud con frase 1
'''
frase_referencia = frases[0]
vector_referencia = vectores[0]

print(f"Frase de referencia: '{frase_referencia}'\n")
print("Similitud con otras frases:")
print("-" * 60)

# Iteramos sobre todas las frases excepto la primera (índice 0)
# enumerate(frases[1:], start=1) nos da el índice original (1, 2, 3, 4)
for i, frase in enumerate(frases[1:], start=1):
    # Calculamos la similitud coseno entre el vector de referencia
    # y el vector de la frase actual
    similitud = similitud_coseno(vector_referencia, vectores[i])
    
    # Mostramos los resultados formateados
    print(f"Frase {i+1}: '{frase}'")
    print(f"Similitud Coseno: {similitud:.4f}")
    print("-" * 60)

# Interpretación de resultados:
# - Frase 2 debería tener similitud > 0.7 (muy similar semánticamente)
# - Frases 3-4 deberían tener similitud ~0.2-0.4 (poco relacionadas)
# - Frase 5 debería tener similitud < 0.2 (muy diferente)


# ==============================================================================
# PASO 6: DEMOSTRACIÓN DE BÚSQUEDA SEMÁNTICA
# ==============================================================================

'''
Paso 6: Demostración de búsqueda semántica

Ahora simulamos un caso de uso real de RAG:
1. El usuario hace una pregunta en lenguaje natural
2. Convertimos la pregunta a un vector de embeddings
3. Comparamos la pregunta con todos los documentos disponibles
4. Devolvemos el documento más relevante

Este es el fundamento de los sistemas RAG:
- En lugar de buscar palabras exactas (búsqueda léxica)
- Buscamos significado similar (búsqueda semántica)

Ventajas sobre búsqueda tradicional:
- "perro" y "canino" se consideran equivalentes
- "parque" y "jardín" se consideran similares
- La pregunta no necesita contener las mismas palabras que la respuesta
'''
pregunta = "Donde esta jugando el animal?"
print(f"\nBuscando respuesta para: '{pregunta}'")

# embed_query es similar a embed_documents pero para una sola pregunta
# Internamente hace lo mismo: tokeniza y pasa por el transformer
vector_pregunta = modelo_embeddings.embed_query(pregunta)

print("\nPuntuaciones de relevancia:")

# Buscamos el fragmento con mayor similitud coseno
mejor_puntuacion = -1
indice_mejor = 0

# Iteramos sobre todos los fragmentos disponibles
for i, frase in enumerate(frases):
    # Calculamos similitud entre la pregunta y cada fragmento
    similitud = similitud_coseno(vector_pregunta, vectores[i])
    
    # Mostramos la similitud de cada fragmento
    print(f"[{similitud:.4f}] {frase}")
    
    # Actualizamos el mejor fragmento si encontramos uno mejor
    if similitud > mejor_puntuacion:
        mejor_puntuacion = similitud
        indice_mejor = i

# Mostramos el resultado final
print(f"\nEl fragmento más relevante es el índice {indice_mejor}:")
print(f"'{frases[indice_mejor]}'")

# Resultado esperado:
# La frase "El perro esta corriendo en el parque" debería ser la más relevante
# porque semánticamente responde a "¿Dónde está jugando el animal?"
# incluso aunque no contenga las palabras exactas "animal" o "jugando"


# ==============================================================================
# RESUMEN Y PRÓXIMOS PASOS
# ==============================================================================

'''
Lo que hemos aprendido:

1. Los embeddings convierten texto en vectores numéricos
2. Textos con significado similar tienen vectores cercanos
3. La similitud coseno mide qué tan similares son dos vectores
4. Podemos usar embeddings para búsqueda semántica en RAG

Próximos pasos en el curso:
- Integrar estos embeddings con un vector store (FAISS, Chroma)
- Crear un retriever que busque automáticamente los fragmentos relevantes
- Construir un pipeline RAG completo con Groq + HuggingFace

Requisitos instalados:
pip install langchain-huggingface sentence-transformers numpy
'''