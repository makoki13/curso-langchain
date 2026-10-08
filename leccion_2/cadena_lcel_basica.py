from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

# Paso 1: Definir el esquema de Pydantic
class PerfilProfesional(BaseModel):
    nombre: str = Field(description="El nombre completo de la persona.")
    cargo: str = Field(description="El puesto o cargo actual.")
    organizacion: str = Field(description="La empresa u organizacion.")

# Paso 2: Inicializar los componentes Runnables
parser = PydanticOutputParser(pydantic_object=PerfilProfesional)

plantilla = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un extractor de perfiles. {instrucciones_formato}"),
        ("human", "Texto: {texto_entrada}")
    ]
)

modelo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

# Paso 3: Construir la cadena utilizando el operador de tuberia
# Aqui es donde ocurre la magia de LCEL. Unimos los componentes en un solo objeto Runnable.
cadena_extraccion = plantilla | modelo | parser

#Paso 4: Ejecutar la cadena con invoke
texto_libre = "Maria Gomez trabaja como Ingeniera de Prompt Senior en la startup NeuralHarness."

# El metodo invoke de la cadena completa se encarga de pasar los datos por todos los pasos.
resultado = cadena_extraccion.invoke({
    "instrucciones_formato": parser.get_format_instructions(),
    "texto_entrada": texto_libre
})

print("Resultado con invoke:")
print(resultado.nombre, "-", resultado.cargo, "en", resultado.organizacion)

'''
Procesamiento en lotes
'''
print("\n--- Demonstracion de Stream ---")
for chunk in cadena_extraccion.stream({
    "instrucciones_formato": parser.get_format_instructions(),
    "texto_entrada": texto_libre
}):
    print(f"Fragmento recibido: {chunk}")

'''
Y lo que es mas impresionante para un arnes de produccion, podemos procesar multiples textos en paralelo usando batch. LangChain se encarga de la concurrencia interna.
'''

print("\n--- Demonstracion de Batch ---")
lista_de_textos = [
    {"instrucciones_formato": parser.get_format_instructions(), "texto_entrada": "Juan Perez es CEO de TechCorp."},
    {"instrucciones_formato": parser.get_format_instructions(), "texto_entrada": "Ana Ruiz lidera el equipo de IA en DataSystems."}
]

resultados_paralelos = cadena_extraccion.batch(lista_de_textos)
for res in resultados_paralelos:
    print(f"Procesado en lote: {res.nombre}")

