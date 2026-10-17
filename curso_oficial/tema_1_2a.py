from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

@tool
def raiz_cuadrada(x: float) -> float:
    """Calcula la raiz cuadrada de un número"""
    return x ** 0.5

'''
@tool("raiz_cuadrada")
def tool1(x: float) -> float:
    return x ** 0.5

@tool("raiz_cuadrada", description="Calcula la raiz cuadrada de un número")
def tool1(x: float) -> float:
    return x ** 0.5
'''

respuesta = raiz_cuadrada.invoke({"x": 467})
#print(respuesta.content) #da error
'''
El error se produce al intentar acceder a respuesta.content. En Python, los objetos de tipo float no poseen un atributo llamado content. Por lo tanto, el interprete lanza un AttributeError indicando que el objeto float no tiene dicho atributo.
'''

#forma correcta (y sencilla)
print(respuesta)

#forma con envoltura
from langchain_core.messages import ToolMessage  # noqa: E402
resultado_crudo = raiz_cuadrada.invoke({"x": 467})
mensaje_herramienta = ToolMessage(
content=str(resultado_crudo),
tool_call_id="identificador_unico_de_la_llamada"
)
print(mensaje_herramienta.content)
#El objeto mensaje_herramienta (clase ToolMessage) sí tiene el atributo content