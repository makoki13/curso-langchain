import logging
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ConfigDict, StrictInt

load_dotenv()

# ──────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paso 1: Esquema Pydantic
# ──────────────────────────────────────────────
class PerfilProfesional(BaseModel):
    model_config = ConfigDict(strict=True) 

    nombre: StrictInt = Field(description="El nombre completo de la persona.")
    edad: int | None = Field(description="La edad en años. Si no se menciona, null.")
    cargo: str = Field(description="El puesto o cargo actual.")
    organizacion: str = Field(description="La empresa u organización donde trabaja.")

# ──────────────────────────────────────────────
# Paso 2: Parser
# ──────────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=PerfilProfesional)

# ──────────────────────────────────────────────
# Paso 3: Plantillas
# ──────────────────────────────────────────────
plantilla_inicial = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en extracción de información de perfiles profesionales. "
            "Tu tarea es analizar el texto proporcionado y extraer los datos solicitados.\n\n"
            "{instrucciones_formato}"
        ),
        (
            "human",
            "Texto a analizar:\n\n{texto_entrada}"
        )
    ]
)

# Plantilla para el reintento: incluye el error para que el modelo se corrija
plantilla_reintento = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en extracción de información de perfiles profesionales. "
            "Tu tarea es analizar el texto proporcionado y extraer los datos solicitados.\n\n"
            "{instrucciones_formato}"
        ),
        (
            "human",
            "Texto a analizar:\n\n{texto_entrada}"
        ),
        (
            "ai",
            "{respuesta_anterior}"
        ),
        (
            "human",
            "Tu respuesta anterior no es válida. Se produjo el siguiente error de validación:\n\n"
            "{error_validacion}\n\n"
            "Por favor, corrige tu respuesta y devuelve ÚNICAMENTE el JSON válido, "
            "sin texto adicional ni bloques de código."
        )
    ]
)

# ──────────────────────────────────────────────
# Paso 4: Modelo
# ──────────────────────────────────────────────
modelo = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0)

# ──────────────────────────────────────────────
# Paso 5: Función auxiliar para extraer texto
# ──────────────────────────────────────────────
def extraer_texto_de_respuesta(mensaje) -> str:
    """Extrae el contenido de texto de un AIMessage, manejando contenido multimodal."""
    if isinstance(mensaje.content, str):
        return mensaje.content
    elif isinstance(mensaje.content, list):
        partes_texto = []
        for parte in mensaje.content:
            if isinstance(parte, str):
                partes_texto.append(parte)
            elif isinstance(parte, dict) and parte.get("type") == "text":
                partes_texto.append(parte.get("text", ""))
        return "\n".join(partes_texto)
    else:
        return str(mensaje.content)

# ──────────────────────────────────────────────
# Paso 6: Función principal con reintentos
# ──────────────────────────────────────────────
MAX_REINTENTOS = 3

def extraer_perfil(texto_entrada: str) -> PerfilProfesional | None:
    """
    Extrae un PerfilProfesional del texto libre.
    Si el parseo falla, reintenta enviando el error al modelo (autorreparación).
    """

    instrucciones = parser.get_format_instructions()

    # ── Primer intento ──
    mensajes = plantilla_inicial.format_messages(
        instrucciones_formato=instrucciones,
        texto_entrada=texto_entrada
    )

    respuesta_cruda = modelo.invoke(mensajes)
    texto_respuesta = extraer_texto_de_respuesta(respuesta_cruda)

    # ── Bucle de reintentos ──
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            perfil = parser.parse(texto_respuesta)
            logger.info("Parseo exitoso en el intento %d.", intento)
            return perfil

        except Exception as e:
            logger.warning(
                "Intento %d/%d falló. Error: %s",
                intento, MAX_REINTENTOS, e
            )

            # Si era el último intento, no reintenta más
            if intento == MAX_REINTENTOS:
                logger.error(
                    "Se agotaron los %d reintentos. "
                    "Última respuesta del modelo:\n%s",
                    MAX_REINTENTOS, texto_respuesta
                )
                return None

            # ── Autorreparación: enviar el error al modelo ──
            logger.info("Reintentando con retroalimentación del error...")

            mensajes_reintento = plantilla_reintento.format_messages(
                instrucciones_formato=instrucciones,
                texto_entrada=texto_entrada,
                respuesta_anterior=texto_respuesta,
                error_validacion=str(e)
            )

            respuesta_cruda = modelo.invoke(mensajes_reintento)
            texto_respuesta = extraer_texto_de_respuesta(respuesta_cruda)

    return None  # No debería llegar aquí, pero por seguridad

# ──────────────────────────────────────────────
# Paso 7: Ejecución
# ──────────────────────────────────────────────
if __name__ == "__main__":
    texto_libre = (
        "En la conferencia de este año, el Dr. Pablo Prieto, "
        "quien se desempeña como CEO de Makoki Enterprises, "
        "presentó una ponencia sobre El LP Revolver de los Beatles."
    )

    perfil = extraer_perfil(texto_libre)

    if perfil is not None:
        print("\n✅ Extracción exitosa:")
        print(f"   Nombre:        {perfil.nombre}")
        print(f"   Edad:          {perfil.edad}")
        print(f"   Cargo:         {perfil.cargo}")
        print(f"   Organización:  {perfil.organizacion}")
        print(f"\n   Como dict: {perfil.model_dump()}")
    else:
        print("\n❌ No se pudo extraer el perfil tras todos los reintentos.")
