"""Revisa las fuentes de la clase 9 con consultas pequeñas.

    python pruebas/fuentes_clase09.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import requests

from taller import clase09_firms, fuentes_clase09


def probar():
    """Devuelve una lista de mensajes. Lanza AssertionError con un mensaje claro si una fuente cambió."""
    mensajes = []

    # MLSUM: que el parquet exista y que el servicio de filas entregue una noticia con las columnas esperadas
    cabeza = requests.head(fuentes_clase09.MLSUM, allow_redirects=True, timeout=60)
    assert cabeza.status_code == 200, f"Hugging Face respondió {cabeza.status_code} para el parquet de MLSUM"
    filas = requests.get("https://datasets-server.huggingface.co/rows", timeout=60,
                         params={"dataset": "mteb/mlsum", "config": "es", "split": "test", "offset": 0, "length": 1})
    assert filas.status_code == 200, f"El servicio de filas de Hugging Face respondió {filas.status_code}"
    fila = filas.json()["rows"][0]["row"]
    for columna in ["text", "summary", "topic", "url", "title", "date"]:
        assert columna in fila, f"MLSUM: falta la columna {columna}"
    assert "elpais.com" in fila["url"], f"MLSUM: la noticia 0 ya no es de El País ({fila['url']})"
    mensajes.append(f"MLSUM es: parquet disponible; noticia 0: «{fila['title'][:50]}»")

    # FIRMS: un año completo y sus columnas
    respuesta = requests.get(fuentes_clase09.FIRMS.format(anio=2023), timeout=120)
    assert respuesta.status_code == 200, f"FIRMS respondió {respuesta.status_code} para 2023"
    focos = pd.read_csv(io.BytesIO(respuesta.content))
    for columna in ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "type"]:
        assert columna in focos.columns, f"FIRMS: falta la columna {columna}"
    assert len(focos) == 22771, f"FIRMS 2023: esperaba 22.771 focos y llegaron {len(focos)} (el archivo histórico cambió)"
    mensajes.append(f"FIRMS MODIS 2023: {len(focos)} focos, de {focos['acq_date'].min()} a {focos['acq_date'].max()}")

    # Límites departamentales
    limites = requests.get(clase09_firms.GEO_URL, timeout=120)
    assert limites.status_code == 200, f"geoBoundaries respondió {limites.status_code}"
    assert len(limites.json()["features"]) == 33, "geoBoundaries: ya no son 33 departamentos"
    mensajes.append("geoBoundaries COL ADM1: 33 polígonos")
    return mensajes


if __name__ == "__main__":
    for mensaje in probar():
        print("OK    ", mensaje)
