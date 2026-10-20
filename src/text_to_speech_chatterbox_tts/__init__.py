import torchaudio as ta 
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
	device="cpu"
)
texto = "Hola, Bienvenidos a la Inteligencia Artificial generativa, vamos  crear audios, imagenes, textos y videos"

print("Generando Audio...")

audio = modelo.generate(
	texto,
	language_id="es",
	exaggeration=0.7, #expresion
	cfg_weight=0.4, #nos ajusta el ritmo si la voz va rapida
	temperature=0.8, #control la variabilidad Mas alta=mas variabilidad, menos alta=menos
	repetition_penalty=1.2,#Penaliza repeticiones extrañas
	min_p=0.05, #filtra opciones improbables durante la generacion
	top_p=1.0 #controla el muestreo probabilistico, 1.0 lo deja abierto
)

ta.save(
	"audio.wav",
	audio,
	modelo.sr
)
print("Audio guardado.")