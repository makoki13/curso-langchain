import logging

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

# Configuramos un logger basico para registrar eventos del arnes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Paso 1: Definir el esquema de datos con validaciones de negocio
class EmpleadoExtraido(BaseModel):
    nombre_completo: str = Field(description="Nombre y apellidos completos de la persona.")
    email: str | None = Field(description="Correo electronico si esta mencionado. Debe tener formato valido.", default=None)
    departamento: str = Field(description="Departamento o area de la empresa donde trabaja o trabajara.")
    habilidades: list[str] = Field(description="Lista de habilidades tecnicas o competencias mencionadas.")
    anios_experiencia: int | None = Field(description="Anios de experiencia mencionados. Null si no se especifica.", default=None)

    # Validador personalizado para el email
    @field_validator('email')
    @classmethod
    def validar_email(cls, v):
        if v is not None and '@' not in v:
            raise ValueError('El email debe contener un arroba.')
        return v

class ResultadoExtraccion(BaseModel):
    empleados_encontrados: list[EmpleadoExtraido] = Field(description="Lista de empleados extraidos del texto.")
    resumen_general: str = Field(description="Un resumen de una frase sobre el contenido del texto.")

# Construccion de los componentes del arnes
# Ahora que tenemos el contrato de datos, debemos construir los componentes que procesaran la informacion.
# Paso 2: Configurar el parser de salida
# El parser utilizara el esquema de Pydantic para generar las instrucciones de formato y validar la salida del modelo.
parser = PydanticOutputParser(pydantic_object=ResultadoExtraccion)

# Paso 3: Construir la plantilla de prompt
# La plantilla debe ser clara, concisa y debe inyectar las instrucciones del parser para garantizar que el modelo devuelva un JSON valido.
plantilla_extraccion = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un sistema experto en extraccion de informacion de recursos humanos. "
            "Tu tarea es analizar el texto proporcionado y extraer todos los datos de los empleados mencionados. "
            "Debes devolver estrictamente un objeto JSON que cumpla con el esquema proporcionado. "
            "Si no encuentras informacion sobre un campo opcional, usalo como null. "
            "Si no encuentras ningun empleado, devuelve una lista vacia.\n\n"
            "{instrucciones_formato}"
        ),
        (
            "human",
            "Texto a analizar:\n\n{texto_documento}"
        )
    ]
)

# Implementacion de la robustez y disponibilidad
'''
En un entorno de produccion, no podemos confiar en que una sola llamada a la API funcione siempre. Vamos a implementar una estrategia de defensa en profundidad.'''

# Paso 4: Inicializar modelos con reintentos y fallbacks
# Modelo principal con reintentos
modelo_principal = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
modelo_principal_resiliente = modelo_principal.with_retry(stop_after_attempt=3)

# Modelo de respaldo
modelo_respaldo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)


# Modelo con fallback configurado
modelo_con_respaldo = modelo_principal_resiliente.with_fallbacks([modelo_respaldo])

# Construccion de la cadena LCEL completa
'''
Ahora unimos todos los componentes mediante el operador de tuberia. Pero hay un detalle importante: queremos registrar en el log cuantos empleados se extrajeron y cuanto tiempo tardo el proceso. Para ello, utilizaremos RunnableLambda para inyectar logica de observabilidad.
'''
#Paso 5: Funcion de observabilidad
def registrar_extraccion(resultado: ResultadoExtraccion) -> ResultadoExtraccion:
    # Esta funcion no modifica el resultado, solo registra metricas utiles.
    num_empleados = len(resultado.empleados_encontrados)
    logging.info(f"Extraccion completada. Se encontraron {num_empleados} empleados.")
    if num_empleados == 0:
        logging.warning("No se encontraron empleados en el texto proporcionado.")
    return resultado

# Paso 6: Ensamblaje de la cadena completa
cadena_extraccion_completa = (
    {
        "instrucciones_formato": lambda _: parser.get_format_instructions(),
        "texto_documento": RunnablePassthrough()
    }
    | plantilla_extraccion
    | modelo_con_respaldo
    | parser
    | RunnableLambda(registrar_extraccion)
)

'''
Observa el primer paso de la cadena. Utilizamos un diccionario para construir la entrada que espera la plantilla. La clave instrucciones_formato utiliza una funcion lambda para obtener las instrucciones del parser, mientras que texto_documento simplemente pasa el texto de entrada sin modificarlo.
'''
# Ejecucion y analisis de resultados
# Vamos a probar nuestro arnes con un texto realista y complejo.
# Paso 7: Ejecucion del arnes
if __name__ == "__main__":
    texto_prueba = """
    En la reunion de ayer, conocimos a nuestro nuevo Director de Tecnologia, Carlos Mendez.
    Carlos tiene 15 anios de experiencia en arquitectura de software y es experto en Python,
    Kubernetes y sistemas distribuidos. Su correo es carlos.mendez@empresa.com.

    Tambien se unio al equipo de Data Science Laura Ramirez, quien trabajara en el departamento
    de Inteligencia Artificial. Laura tiene 5 anios de experiencia y domina TensorFlow, PyTorch
    y MLOps. Puedes contactarla en laura.r@empresa.com.

    Adicionalmente, mencionaron que Pedro Sanchez se mudara al departamento de Ventas, aunque
    no se especificaron sus habilidades tecnicas ni su correo.
    """

    try:
        logging.info("Iniciando proceso de extraccion...")
        resultado_final = cadena_extraccion_completa.invoke(texto_prueba)

        print("\n=== RESULTADO DE LA EXTRACCION ===")
        print(f"Resumen: {resultado_final.resumen_general}")
        print(f"\nEmpleados encontrados: {len(resultado_final.empleados_encontrados)}")

        for i, empleado in enumerate(resultado_final.empleados_encontrados, 1):
            print(f"\n--- Empleado {i} ---")
            print(f"Nombre: {empleado.nombre_completo}")
            print(f"Email: {empleado.email}")
            print(f"Departamento: {empleado.departamento}")
            print(f"Habilidades: {', '.join(empleado.habilidades)}")
            print(f"Anios de experiencia: {empleado.anios_experiencia}")

    except Exception as e:
        logging.error(f"El arnes fallo despues de todos los reintentos y fallbacks: {e}")
        print("Error critico en el sistema de extraccion. Por favor, contacte a soporte.")
