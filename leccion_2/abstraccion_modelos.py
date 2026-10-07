from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

'''Esta funcion representa la logica de negocio de nuestro arnes.
Observa que no sabe ni le importa que modelo esta usando.
Solo sabe que recibe un objeto que implementa la interfaz BaseChatModel.'''

def analizar_sentimiento_de_producto(modelo_de_lenguaje, texto_producto):
    print(f"\n--- Ejecutando analisis con {type(modelo_de_lenguaje).__name__} ---")

    mensajes = [
        SystemMessage(content="Eres un analista de mercado experto. Responde de forma concisa y profesional."),
        HumanMessage(content=f"Analiza el sentimiento y extrae la caracteristica principal del siguiente producto: {texto_producto}")
    ]

    # Invoke es el metodo estandar de la interfaz Runnable.
    respuesta = modelo_de_lenguaje.invoke(mensajes)

    print(f"Respuesta: {respuesta.content}")
    print(f"Tokens utilizados: {respuesta.usage_metadata}")

if __name__ == "__main__":
    texto_a_analizar = "La nueva bateria del telefono dura 48 horas, pero la pantalla se raya con solo mirarla."

    # Inicializamos el primer proveedor: OpenAI
    nombre_modelo = "qwen/qwen3.8-27b"
    modelo_groq = ChatGroq(
    model=nombre_modelo,
    temperature=0.0
)

    # Inicializamos el segundo proveedor: Anthropic
    modelo_anthropic = ChatAnthropic(
        model_name="claude-3-haiku-20240307",  # Cambiado de model_name a model
        temperature=0.0,
        timeout=60,
        stop=None,
        max_tokens_to_sample=150  # Cambiado de max_tokens_to_sample a max_tokens
    )

    # Ejecutamos la EXACTA misma logica de negocio con ambos motores
    analizar_sentimiento_de_producto(modelo_groq, texto_a_analizar)
    analizar_sentimiento_de_producto(modelo_anthropic, texto_a_analizar)
