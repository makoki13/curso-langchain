# pip install ragas datasets

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,   # ← import clave
    RunnableLambda,     # ← import clave
)
from operator import itemgetter # ← import clave para extraer claves
#from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset  # ← nuevo
from ragas.metrics import Faithfulness, AnswerRelevancy  # ← ahora son clases


load_dotenv()

# ── Paso 1: Conectar con el almacen vectorial ─────────────
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
almacen_vectorial = Chroma(
    collection_name="manuals_empresa",
    embedding_function=modelo_embeddings,
    persist_directory="./chroma_db_local"
)
retriever_base = almacen_vectorial.as_retriever(search_kwargs={"k": 3})

# ── Paso 2: Implementar la tecnica HyDE con LCEL ──────────
prompt_hyde = ChatPromptTemplate.from_template(
    "Dada la siguiente pregunta, genera una respuesta hipotetica detallada y plausible. "
    "No te preocupes si la respuesta es factualmente correcta, solo genera un texto en el mismo estilo "
    "que los documentos de un manual de empresa.\n\n"
    "Pregunta: {pregunta}"
)

modelo_llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

retriever_hyde = prompt_hyde | modelo_llm | StrOutputParser() | retriever_base

# ── Paso 3: Construir el pipeline RAG ─────────────────────
def formatear_documentos(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# Envolvemos la función para que el type-checker la reconozca como Runnable
formatear = RunnableLambda(formatear_documentos)

prompt_rag = ChatPromptTemplate.from_template(
    "Responde a la pregunta utilizando EXCLUSIVAMENTE el contexto proporcionado.\n\n"
    "CONTEXTO:\n{contexto}\n\n"
    "Pregunta: {pregunta}"
)

# Función auxiliar para convertir el string de entrada en el dict que espera el prompt
def a_dict_pregunta(texto: str) -> dict:
    return {"pregunta": texto}

cadena_rag_hyde = (
    # 1. Convertimos el string de entrada en un dict {"pregunta": "..."}
    RunnableLambda(a_dict_pregunta)
    # 2. Ejecutamos en paralelo y reestructuramos la salida
    | RunnableParallel(
        contexto=retriever_hyde | formatear,
        pregunta=itemgetter("pregunta")  # Extraemos el string original para prompt_rag
    )
    | prompt_rag
    | modelo_llm
    | StrOutputParser()
)

# ── Paso 4: Preparar el dataset para evaluacion ───────────
preguntas_eval = [
    "Cuantos dias de vacaciones tengo al ano?",
    "Que formulario necesito para un reembolso de viaje?"
]

respuestas_generadas = []
contextos_recuperados = []
verdad_fundamental = [
    "Los empleados tienen derecho a 22 dias laborables de vacaciones al ano.",
    "Se debe utilizar el formulario F-104 y adjuntar facturas."
]

print("Generando respuestas y recuperando contextos para la evaluacion...")
for pregunta in preguntas_eval:
    # Generar la respuesta (la cadena ahora acepta un string gracias a a_dict_pregunta)
    respuesta = cadena_rag_hyde.invoke(pregunta)
    respuestas_generadas.append(respuesta)
    
    # Recuperar los documentos (retriever_hyde espera un dict, se lo pasamos explícitamente)
    docs = retriever_hyde.invoke({"pregunta": pregunta})
    contextos_recuperados.append([doc.page_content for doc in docs])

# ── Paso 5: Ejecutar la evaluacion con RAGAS ─  (adaptado a ragas ≥ 0.2) 
# Construir el dataset con la nueva API
eval_dataset = EvaluationDataset.from_dict({
    "user_input": preguntas_eval,          # antes: "question"
    "response": respuestas_generadas,      # antes: "answer"
    "retrieved_contexts": contextos_recuperados,  # antes: "contexts"
    "reference": verdad_fundamental,       # antes: "ground_truth"
})

print("\nEjecutando evaluacion con RAGAS...")
resultados = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],  # ← se instancian
)

print("\n--- RESULTADOS DE LA EVALUACION ---")
print(resultados)