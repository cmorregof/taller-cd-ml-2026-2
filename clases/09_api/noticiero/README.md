# El noticiero de un minuto

Demostración de la clase 9 · Taller en Ciencia de Datos y Aprendizaje Automático · UNAL Manizales · 2026-II

Una noticia real de El País (corpus MLSUM), resumida por Claude y leída por una voz de ElevenLabs.

```
noticiero/
├── api/            La API, con FastAPI. Se despliega en Railway.
│   ├── main.py
│   ├── requirements.txt
│   ├── Procfile
│   └── .env.ejemplo
└── web/            La página, un solo HTML. Se despliega en Vercel.
    ├── index.html
    ├── config.js   La dirección de la API: la única línea que se cambia al desplegar.
    └── vercel.json
```

## Correrlo en el computador

```bash
cd api
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # sin las llaves la API arranca, pero /resumen y /voz responden 503
export ELEVENLABS_API_KEY=...
uvicorn main:app --reload           # la API queda en http://127.0.0.1:8000 y su documentación en /docs
```

En otra terminal:

```bash
cd web
python -m http.server 8001          # la página queda en http://127.0.0.1:8001
```

## Rutas

| Ruta | Qué hace | Necesita llave |
|---|---|---|
| `GET /` | Estado del servicio y llaves configuradas | No |
| `GET /noticia/{indice}` | La noticia número `indice`, de 0 a 13.919 | No |
| `POST /resumen/{indice}` | Resumen de un minuto, con duración estimada y control de cifras | Claude |
| `POST /voz` | Recibe `{"texto": "..."}` y devuelve un MP3 | ElevenLabs |

## Variables de entorno

| Variable | Para qué | Valor por defecto |
|---|---|---|
| `ANTHROPIC_API_KEY` | Llave de la API de Claude | Ninguno |
| `ELEVENLABS_API_KEY` | Llave de ElevenLabs | Ninguno |
| `CLAUDE_MODELO` | Modelo que resume | `claude-haiku-5-5` |
| `ELEVENLABS_VOZ` | Identificador de la voz | `JBFqnCBsd6RMkjVDRZzb`, la voz del ejemplo de la documentación |
| `ELEVENLABS_MODELO` | Modelo de voz | `eleven_multilingual_v2` |
| `ORIGENES` | Direcciones de la página autorizadas a llamar a la API, separadas por comas | `*` |
| `CLAVE_DEL_SALON` | Si existe, `/resumen` y `/voz` exigen el encabezado `X-Clave` con ese valor | Ninguno |

Las llaves no se escriben en el código ni se suben al repositorio. En Railway van en la pestaña de variables del servicio.

## Desplegar

**API en Railway.** Proyecto nuevo desde el repositorio de GitHub, carpeta raíz `api`, las dos llaves en las variables, y generar un dominio público. La orden de arranque está en `Procfile`.

**Página en Vercel.** Cambiar la dirección en `web/config.js` por la de Railway, importar el repositorio en Vercel con carpeta raíz `web` y sin paso de compilación. Después, poner la dirección de Vercel en la variable `ORIGENES` de Railway.

## Qué se probó y qué no

Probado el 18 de noviembre de 2026, sin llaves: la API con el cliente de pruebas de FastAPI (estado, noticia, errores 404, 422 y 503), y la página en un navegador contra la API local, hasta el mensaje de «falta la llave».

**No probado**, porque exige llaves y cuentas: la llamada real a Claude, la llamada real a ElevenLabs, el despliegue en Railway y el despliegue en Vercel. Los nombres de modelos y los parámetros se tomaron de la documentación oficial de Anthropic y de ElevenLabs consultada ese mismo día.

## Datos y derechos

Las noticias son de El País y llegan desde el corpus MLSUM en Hugging Face (`mteb/mlsum`, configuración `es`), publicado para investigación. Este servicio no guarda ni republica los textos: los pide uno por uno y enlaza la noticia original. Es un ejercicio de clase.
