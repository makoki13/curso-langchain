from langchain_groq import ChatGroq  # noqa: E402
from langchain.messages import HumanMessage, SystemMessage


system_prompt_text = "You are a science fiction writer, create a capital city at the users request."
system_prompt=SystemMessage(content=system_prompt_text),
human_question_text="What's the capital of the moon?"
human_question = HumanMessage(content=human_question_text)


scifi_agent = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.0
    )

mensajes = [
    system_prompt,
    human_question
]

response = scifi_agent.invoke(mensajes)

print(response.content)