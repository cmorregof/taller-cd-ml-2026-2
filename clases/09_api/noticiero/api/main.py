"""El noticiero de un minuto: una noticia real, resumida por Claude y leída por ElevenLabs.

Para correrla en el computador:

    pip install -r requirements.txt
    uvicorn main:app --reload

Las llaves NUNCA van en el código. Se leen de variables de entorno:

    ANTHROPIC_API_KEY     llave de la API de Claude
    ELEVENLABS_API_KEY    llave de ElevenLabs
    CLAUDE_MODELO         modelo de Claude (por defecto, claude-haiku-5-5)
    ELEVENLABS_VOZ        identificador de la voz (por defecto, la del ejemplo de la documentación)
    ELEVENLABS_MODELO     modelo de voz (por defecto, eleven_multilingual_v2)
    ORIGENES              direcciones de la página que pueden llamar a la API, separadas por comas (por defecto, *)
    CLAVE_DEL_SALON       opcional; si existe, /resumen y /voz exigen el encabezado X-Clave con ese valor

Sin llaves, la API arranca y responde 503 en lo que las necesita. No inventa respuestas.
"""
from __future__ import annotations

import os
import re

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

FILAS = "https://datasets-server.huggingface.co/rows"       # una noticia de MLSUM por consulta, sin bajar el corpus
TOTAL_NOTICIAS = 13920                                      # partición de prueba de MLSUM en español
PALABRAS_POR_MINUTO = 150                                   # supuesto de locución; se contrasta con el audio real
PALABRAS_OBJETIVO = 140                                     # un poco menos de un minuto
MAX_CARACTERES_VOZ = 1500                                   # tope para que nadie convierta un libro con nuestra llave

MODELO_CLAUDE = os.environ.get("CLAUDE_MODELO", "claude-haiku-5-5")
VOZ = os.environ.get("ELEVENLABS_VOZ", "JBFqnCBsd6RMkjVDRZzb")
MODELO_VOZ = os.environ.get("ELEVENLABS_MODELO", "eleven_multilingual_v2")

INSTRUCCIONES = (
    "Eres redactor de un noticiero de radio en español. Recibes una noticia y escribes el texto que "
    f"leerá el locutor: un solo párrafo de unas {PALABRAS_OBJETIVO} palabras, con frases cortas, para ser oído y no leído. "
    "Usa solo hechos, nombres y cifras que estén en la noticia. No agregues opiniones ni contexto que no esté en ella. "
    "Si la noticia no da un dato, no lo supongas. Escribe los números como se dicen en voz alta. "
    "No uses títulos, viñetas ni saludos: entrega únicamente el párrafo."
)

app = FastAPI(title="El noticiero de un minuto", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ORIGENES", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])

_noticias: dict[int, dict] = {}      # memoria de la sesión: no se le pide dos veces lo mismo a nadie
_resumenes: dict[int, dict] = {}
_audios: dict[str, bytes] = {}


# ------------------------------------------------------------------ piezas que se pueden probar sin llaves
def contar_palabras(texto: str) -> int:
    return len(texto.split())


def segundos_estimados(texto: str) -> float:
    """Duración al aire si se lee a PALABRAS_POR_MINUTO."""
    return round(contar_palabras(texto) / PALABRAS_POR_MINUTO * 60, 1)


def cifras_sin_respaldo(resumen: str, noticia: str) -> list[str]:
    """Números escritos con dígitos en el resumen que no aparecen en la noticia.

    Es un control barato y no una garantía: no ve los números escritos con letras
    ni los nombres cambiados. Una lista vacía no prueba que el resumen sea fiel.
    """
    normalizar = lambda numero: numero.replace(".", "").replace(",", ".").rstrip(".")
    en_noticia = {normalizar(n) for n in re.findall(r"\d[\d.,]*", noticia)}
    return sorted({n for n in re.findall(r"\d[\d.,]*", resumen) if normalizar(n) not in en_noticia})


def traer_noticia(indice: int) -> dict:
    """Trae una noticia de El País desde el corpus MLSUM (Hugging Face), por su posición."""
    if not 0 <= indice < TOTAL_NOTICIAS:
        raise HTTPException(404, f"No hay noticia {indice}. Los índices van de 0 a {TOTAL_NOTICIAS - 1}.")
    if indice not in _noticias:
        try:
            respuesta = requests.get(FILAS, timeout=30, params={"dataset": "mteb/mlsum", "config": "es", "split": "test",
                                                               "offset": indice, "length": 1})
            respuesta.raise_for_status()
            fila = respuesta.json()["rows"][0]["row"]
        except Exception as error:
            raise HTTPException(502, f"Hugging Face no entregó la noticia {indice}: {type(error).__name__}.") from error
        _noticias[indice] = {"indice": indice, "titulo": fila["title"], "fecha": fila["date"], "tema": fila["topic"],
                             "url": fila["url"], "entradilla": fila["summary"], "texto": fila["text"],
                             "palabras": contar_palabras(fila["text"])}
    return _noticias[indice]


def _revisar_clave(clave: str | None) -> None:
    esperada = os.environ.get("CLAVE_DEL_SALON")
    if esperada and clave != esperada:
        raise HTTPException(401, "Falta el encabezado X-Clave o no coincide.")


# ------------------------------------------------------------------ piezas que necesitan llave
def resumir_con_claude(noticia: dict) -> str:
    llave = os.environ.get("ANTHROPIC_API_KEY")
    if not llave:
        raise HTTPException(503, "Falta la variable de entorno ANTHROPIC_API_KEY: sin ella no hay resumen.")
    import anthropic

    try:
        mensaje = anthropic.Anthropic(api_key=llave).messages.create(
            model=MODELO_CLAUDE,
            max_tokens=600,
            system=INSTRUCCIONES,
            messages=[{"role": "user", "content": f"Título: {noticia['titulo']}\n\nNoticia:\n{noticia['texto'][:20000]}"}],
        )
    except anthropic.APIError as error:
        raise HTTPException(502, f"La API de Claude respondió con error: {type(error).__name__}.") from error
    return "".join(bloque.text for bloque in mensaje.content if bloque.type == "text").strip()


def voz_con_elevenlabs(texto: str) -> bytes:
    llave = os.environ.get("ELEVENLABS_API_KEY")
    if not llave:
        raise HTTPException(503, "Falta la variable de entorno ELEVENLABS_API_KEY: sin ella no hay voz.")
    respuesta = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOZ}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": llave, "Content-Type": "application/json"},
        json={"text": texto, "model_id": MODELO_VOZ},
        timeout=120,
    )
    if respuesta.status_code != 200:
        raise HTTPException(502, f"ElevenLabs respondió {respuesta.status_code}.")
    return respuesta.content


# ------------------------------------------------------------------ rutas
class Texto(BaseModel):
    texto: str = Field(min_length=1, max_length=MAX_CARACTERES_VOZ)


@app.get("/")
def estado() -> dict:
    return {"servicio": "El noticiero de un minuto", "estado": "ok", "noticias": TOTAL_NOTICIAS,
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")), "voz": bool(os.environ.get("ELEVENLABS_API_KEY")),
            "modelo": MODELO_CLAUDE}


@app.get("/noticia/{indice}")
def noticia(indice: int) -> dict:
    return traer_noticia(indice)


@app.post("/resumen/{indice}")
def resumen(indice: int, x_clave: str | None = Header(default=None)) -> dict:
    _revisar_clave(x_clave)
    original = traer_noticia(indice)
    if indice not in _resumenes:
        texto = resumir_con_claude(original)
        sospechosas = cifras_sin_respaldo(texto, original["texto"])
        duracion = segundos_estimados(texto)
        _resumenes[indice] = {
            "indice": indice, "titulo": original["titulo"], "url": original["url"], "resumen": texto,
            "palabras": contar_palabras(texto), "segundos_estimados": duracion,
            "cifras_sin_respaldo": sospechosas, "modelo": MODELO_CLAUDE,
            "sale_al_aire": (not sospechosas) and 45 <= duracion <= 70,
        }
    return _resumenes[indice]


@app.post("/voz")
def voz(cuerpo: Texto, x_clave: str | None = Header(default=None)) -> Response:
    _revisar_clave(x_clave)
    if cuerpo.texto not in _audios:
        _audios[cuerpo.texto] = voz_con_elevenlabs(cuerpo.texto)
    return Response(content=_audios[cuerpo.texto], media_type="audio/mpeg")
