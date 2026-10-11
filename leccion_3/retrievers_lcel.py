from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,      # ← import clave
    RunnableLambda,        # ← import clave
)

load_dotenv()

# ── Paso 1: Almacén vectorial ──────────────────────────────
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
almacen_vectorial = Chroma(
    collection_name="manuals_empresa",
    embedding_function=modelo_embeddings,
    persist_directory="./chroma_db_local",
)

# ── Paso 2: Retrievers ─────────────────────────────────────
retriever_basico = almacen_vectorial.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
retriever_umbral = almacen_vectorial.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8, "k": 3},
)
retriever_mmr = almacen_vectorial.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5},
)

# ── Paso 3: Función auxiliar ──────────────────────────────
def formatear_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# La envolvemos explícitamente para que el type-checker
# reconozca que es un Runnable y no una función suelta.
formatear = RunnableLambda(formatear_docs)

# ── Paso 4: Cadenas con RunnableParallel (explícito) ──────
cadena_basica = RunnableParallel(
    contexto=retriever_basico | formatear,
    pregunta=RunnablePassthrough(),
)

cadena_umbral = RunnableParallel(
    contexto=retriever_umbral | formatear,
    pregunta=RunnablePassthrough(),
)

cadena_mmr = RunnableParallel(
    contexto=retriever_mmr | formatear,
    pregunta=RunnablePassthrough(),
)

# ── Paso 5: Ejecutar y comparar ───────────────────────────
pregunta_usuario = "Cuales son las politicas de vacaciones y reembolsos?"

print("=" * 60)
print("1. RETRIEVER BASICO (Similitud, k=3)")
print("=" * 60)
resultado_basico = cadena_basica.invoke(pregunta_usuario)
print(f"Documentos recuperados:\n{resultado_basico['contexto']}\n")

print("=" * 60)
print("2. RETRIEVER CON UMBRAL (Score > 0.8)")
print("=" * 60)
resultado_umbral = cadena_umbral.invoke(pregunta_usuario)
if not resultado_umbral["contexto"]:
    print("No se recuperaron documentos que superen el umbral.\n")
else:
    print(f"Documentos recuperados:\n{resultado_umbral['contexto']}\n")

print("=" * 60)
print("3. RETRIEVER MMR (Max Marginal Relevance)")
print("=" * 60)
resultado_mmr = cadena_mmr.invoke(pregunta_usuario)
print(f"Documentos recuperados:\n{resultado_mmr['contexto']}\n")