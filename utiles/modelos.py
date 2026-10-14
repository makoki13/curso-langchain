def get_modelo(nombre: str):
    """
    Devuelve una instancia del modelo de lenguaje indicado.

    Parámetros:
        nombre: "" o "groq" → Groq por defecto.
                "openai" o "chatgpt" → OpenAI.
    """
    nombre = nombre.strip().lower()

    if nombre == "" or nombre == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )

    elif nombre in ("openai", "chatgpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0
        )

    else:
        raise ValueError(f"Modelo no soportado: '{nombre}'")

