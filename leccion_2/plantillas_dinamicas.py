from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

#Definimos la plantilla de chat. Utilizamos tuplas donde el primer elemento es el rol y el segundo es el contenido con variables entre llaves.
plantilla = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en análisis de productos de tecnología. "
            "Tu objetivo es extraer información estructurada de las descripciones de productos. "
            "Reglas estrictas: "
            "1. Si no encuentras la información, responde 'No disponible'. "
            "2. No inventes características que no estén explícitamente en el texto. "
            "3. Mantén un tono profesional y objetivo."
        ),
        (
            "human",
            "A continuación te proporciono la descripción de un producto:\n\n{descripcion_producto}\n\n"
            "Por favor, extrae el nombre del producto, la marca, el precio si está mencionado, "
            "y una lista de las características técnicas principales."
        ),
    ]
)

'''
Una vez definida la plantilla, podemos inspeccionar sus variables. Esto es muy útil para validar que nuestra plantilla espera exactamente los datos que nuestro arnés va a proporcionarle.
'''

print("Variables esperadas por la plantilla:", plantilla.input_variables)

#Para generar los mensajes finales, invocamos el método format_messages, pasando los valores reales para las variables definidas.
descripcion_real = (
    "El nuevo portátil XPS 15 de Dell cuenta con un procesador Intel Core i9 de 13ª generación, "
    "32GB de RAM y una tarjeta gráfica NVIDIA RTX 4070. Su precio de lanzamiento es de 2499 euros. "
    "La pantalla es un panel OLED de 15.6 pulgadas con resolución 4K."
)

mensajes_finales = plantilla.format_messages(descripcion_producto=descripcion_real)

'''
Ahora, mensajes_finales es una lista de objetos SystemMessage y HumanMessage, listos para ser enviados a cualquier modelo de chat. Podemos imprimir el contenido para ver cómo la plantilla ha inyectado el contexto dinámico.
'''

for mensaje in mensajes_finales:
    print(f"Rol: {mensaje.type}")
    print(f"Contenido: {mensaje.content}")
    print("-" * 40)

#Si queremos ejecutar esto contra un modelo, simplemente pasamos la lista de mensajes al método invoke.
modelo = ChatGroq(model="qwen/qwen3.8-27b",temperature=0.0)

respuesta = modelo.invoke(mensajes_finales)
print("\nRespuesta del modelo:\n", respuesta.content)
