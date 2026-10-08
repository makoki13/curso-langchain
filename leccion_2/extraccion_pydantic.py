from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

# Paso 1: Definir el esquema de datos con Pydantic
class PerfilProfesional(BaseModel):
    nombre: str = Field(description="El nombre completo de la persona.")
    edad: int | None = Field(description="La edad de la persona en años. Si no se menciona, dejar como null.")
    cargo: str = Field(description="El puesto o cargo actual de la persona.")
    organizacion: str = Field(description="El nombre de la empresa u organización donde trabaja.")

# Paso 2: Configurar el parser de salida
parser = PydanticOutputParser(pydantic_object=PerfilProfesional)

# Paso 3: Construir la plantilla de prompt
plantilla = ChatPromptTemplate.from_messages(
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

# Paso 4: Inicializar el modelo
modelo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

# Paso 5: Ejecutar el flujo completo
texto_libre = (
    "En la conferencia de este año, el Dr. Alejandro Martínez, de 45 años, "
    "quien se desempeña como Director de Inteligencia Artificial en la empresa "
    "TechSolutions Global, presentó una ponencia sobre arneses de IA."
)

mensajes = plantilla.format_messages(
    instrucciones_formato=parser.get_format_instructions(),
    texto_entrada=texto_libre
)

respuesta_cruda = modelo.invoke(mensajes)

# Paso 6: Parsear y validar la salida
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

# Parseo seguro con manejo de errores
perfil_estructurado = None

try:
    texto_limpio = extraer_texto_de_respuesta(respuesta_cruda)
    perfil_estructurado = parser.parse(texto_limpio)
except Exception as e:
    print(f"Error al parsear la respuesta: {e}")

# Verificación antes de acceder a los atributos
if perfil_estructurado is not None:
    print("Extracción exitosa:")
    print(f"Nombre: {perfil_estructurado.nombre}")
    print(f"Edad: {perfil_estructurado.edad}")
    print(f"Cargo: {perfil_estructurado.cargo}")
    print(f"Organización: {perfil_estructurado.organizacion}")
    print("\nComo diccionario:", perfil_estructurado.model_dump())
else:
    print("No se pudo extraer el perfil de la respuesta del modelo.")
