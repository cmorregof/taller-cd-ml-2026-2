"""Revisa las fuentes de la clase 5 con consultas pequeñas (no baja las fotos).

    python pruebas/fuentes_clase05.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taller import fuentes_clase05 as f
from taller.clase05_imagenes import ZipRemoto, pedir, reducir


def _ham10000():
    tabla = f.ham_metadatos()
    esperadas = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]
    assert list(tabla.columns) == esperadas, f"HAM10000: las columnas ahora son {list(tabla.columns)}"
    assert len(tabla) == 10_015, f"HAM10000: había 10.015 imágenes y ahora hay {len(tabla)}"
    muestra = tabla[f.ham_en_muestra(tabla)]
    assert len(muestra) == 3_611, f"HAM10000: el subconjunto del curso tenía 3.611 imágenes y ahora tiene {len(muestra)}"
    parte = ZipRemoto(f.DATAVERSE.format(f.HAM_PARTES[0]))
    assert len(parte.indice) == 5_000, f"HAM10000: la parte 1 tenía 5.000 fotos y ahora tiene {len(parte.indice)}"
    nombre = next(n for n in muestra["image_id"] + ".jpg" if n in parte.indice)
    foto = reducir(parte.leer(nombre))
    assert foto[:2] == b"\xff\xd8", "HAM10000: lo que llegó del ZIP remoto no es un JPEG"
    return f"HAM10000 (Dataverse): 10.015 filas, subconjunto de 3.611, lectura por rangos del ZIP correcta ({nombre})"


def _rocole():
    listado = f.rocole_listado()
    fotos = [x for x in listado if x["filename"].lower().endswith(".jpg")]
    assert len(fotos) == 1_560, f"RoCoLe: había 1.560 fotos y ahora hay {len(fotos)}"
    assert all(f.PATRON.match(x["filename"]) for x in fotos), "RoCoLe: algún nombre de foto ya no sigue el patrón C..P..letras"
    etiquetas = f.rocole_etiquetas(listado)
    assert len(etiquetas) == 1_560, f"RoCoLe: las etiquetas tenían 1.560 filas y ahora tienen {len(etiquetas)}"
    assert set(etiquetas["Binary.Label"]) == {"healthy", "unhealthy"}, "RoCoLe: cambiaron los valores de Binary.Label"
    foto = reducir(pedir(fotos[0]["content_details"]["download_url"]).content)
    assert foto[:2] == b"\xff\xd8", "RoCoLe: lo que llegó no es un JPEG"
    return "RoCoLe (Mendeley Data): 1.560 fotos, 1.560 etiquetas, descarga de una foto correcta"


def probar():
    return [_ham10000(), _rocole()]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
