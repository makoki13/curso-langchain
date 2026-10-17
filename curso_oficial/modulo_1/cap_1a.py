from dotenv import load_dotenv
from langchain_groq import ChatGroq  # noqa: E402
from pprint import pprint  # noqa: E402

load_dotenv()

#####Inicializando e invocando un modelo

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
response = model.invoke("¿Cual es la capital de la luna?")
print(response.content)
print('------------ Metada -----------------')
pprint(response.response_metadata)




