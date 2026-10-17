from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from groq import Groq

ruta_audio = "curso_oficial\\modulo_1\\test.ogg"
print(f"Cargando audio desde {ruta_audio}...")
with open(ruta_audio, "rb") as fichero_audio:
    audio_bytes = fichero_audio.read()
    print("Audio cargado correctamente en memoria.")

""" aud_b64 = base64.b64encode(audio_bytes).decode("utf-8")
multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this audio file"},
    {"type": "audio", "base64": aud_b64, "mime_type": "audio/ogg"}
    ])

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0
)
agent = create_agent(model=model) """

cliente_groq = Groq()
transcripcion = cliente_groq.audio.transcriptions.create(
    file=("audio.ogg", audio_bytes),
    model="whisper-large-v3",
    response_format="text",
)

print(f"Transcripcion completada: {transcripcion}")
multimodal_question = HumanMessage(content=f"Tell me about this audio transcript: {transcripcion}")

model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
agent = create_agent(model=model)

response = agent.invoke(
{"messages": [multimodal_question]}
)
print("Respuesta del modelo:")
print(response['messages'][-1].content)

# Texto generado por el modelo
'''
Transcripcion completada:  Hola Itana, me alegra que te haya gustado, mola mogollón y además si es el primer día y ya te ha gustado vas a flipar porque cuando ya conozcas a más gente verás que buen rollo hay y hay un montón de juegos y muy chulo te dan una acampada pronto de inicio de ronda, estos se van el fin de semana que viene Entonces, nada, cuando llegue el momento de irte de acampada No lo dudes, que se pasa súper, súper bien Y nada, esos juegos no son nada a mí también Pues nada Lo de la pañueleta no te lo aconsejo que lo robes Pero bueno, esto creo que te lo tienes que ganar El tema de la pañueleta no es ahora, ¿vale? Es algo que te tienes que ir ganando Como buena scout Y cuando llegue el momento, pues hace un se hace un ritual y te ponen la pañoleta y nada, es todo muy chulo me alegro un montón, a la semana que viene más hazle un besito
Respuesta del modelo:
This audio transcript is a casual, friendly conversation in **Spanish** between two people. The speaker is addressing someone named **Itana** (likely a young girl or new member) and giving her advice and encouragement about her experience in a **Scout group** (or similar youth organization).

Here’s a breakdown of the key points:

### 1. **Positive Feedback & Encouragement**
- The speaker is happy that Itana enjoyed her first day.
- They emphasize that the group has great vibes (“buen rollo”), lots of fun games, and a welcoming atmosphere.
- They predict she’ll love it even more once she gets to know more people.

### 2. **Upcoming Campout**
- There’s an upcoming campout (“acampada”) at the start of the round/season.
- The current group is leaving for their campout the following weekend.
- The speaker strongly encourages Itana to join the campout when her turn comes, saying it’s “super, super fun.”

### 3. **The Scarf (“Pañuelo”) Tradition**
- Itana seems to have asked about getting a scarf (a common symbol in Scout groups, often representing rank or belonging).
- The speaker gently corrects her: **you can’t just take or steal the scarf**.
- It’s something you **earn** through participation and commitment (“como buena scout” – as a good scout).
- When the time comes, there’s a **ritual** where the scarf is officially given to her.
- The speaker describes this moment as “very cool” (“muy chulo”).

### 4. **Closing**
- The speaker expresses great happiness for Itana.
- They say goodbye warmly, telling her to “give a little kiss” (“hazle un besito”) – a common affectionate sign-off in Spanish-speaking cultures, especially among friends or family.

---

### Cultural & Contextual Notes:
- **“Mola mogollón”**: Very colloquial Spanish (especially from Spain) meaning “it’s awesome” or “I love it.”
- **“Flipar”**: Slang for “to be amazed” or “to be blown away.”
- **“Buen rollo”**: A very common Spanish expression meaning “good vibes” or “positive atmosphere.”
- **Scout Context**: In many Spanish-speaking countries, Scout groups (like *Escultismo* or *Guías y Scouts*) use scarves, badges, and rituals to mark progression and belonging. The scarf is a significant symbol, not just an accessory.

### Tone:
Warm, enthusiastic, supportive, and informal. The speaker is clearly an older member or leader who cares about Itana’s experience and wants to set the right expectations while keeping the mood light and fun.
'''