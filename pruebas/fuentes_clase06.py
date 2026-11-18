"""Revisa las fuentes de la clase 6 con consultas pequeñas.

    python pruebas/fuentes_clase06.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from taller import fuentes_clase06 as f
from taller.clase05_imagenes import pedir, reducir


def _eurosat():
    encabezado = pedir(f.EUROSAT_ZIP, headers={"Range": "bytes=0-3"})
    tamano = int(encabezado.headers["Content-Range"].rsplit("/", 1)[1])
    assert encabezado.content == b"PK\x03\x04", "EuroSAT: lo que hay en la dirección ya no es un ZIP"
    assert tamano == 94_280_567, f"EuroSAT: el ZIP pesaba 94.280.567 bytes y ahora pesa {tamano}"
    return f"EuroSAT (Hugging Face, torchgeo): ZIP de {tamano / 1e6:.1f} MB en su sitio"


def _caldas():
    x0, y1 = f.ZONAS["cafetera"]
    color = f.recorte_pc("sentinel-2-l2a", f.ESCENA, "visual", x0, y1, 64)
    assert color.shape == (4, 64, 64) and color[:3].max() > 0, f"Sentinel-2: el recorte llegó con forma {color.shape} o vacío"
    cobertura = f.recorte_pc("esa-worldcover", f.WORLDCOVER, "map", x0, y1, 64, vecino=True)[0]
    desconocidos = set(np.unique(cobertura).tolist()) - set(f.LEYENDA)
    assert not desconocidos, f"WorldCover: aparecen códigos fuera de la leyenda: {desconocidos}"
    nubes = f.recorte_pc("sentinel-2-l2a", f.ESCENA, "SCL", x0, y1, 64, vecino=True)[0]
    assert nubes.max() <= 11, "Sentinel-2: la capa SCL trae valores fuera de 0 a 11"
    return f"Planetary Computer: escena {f.ESCENA[:26]}…, WorldCover y SCL responden con recortes de 64 x 64"


def _orinoquia():
    metadatos = f.orinoquia_metadatos()
    assert len(metadatos) >= 112_000, f"Orinoquía: había más de 112.000 anotaciones y ahora hay {len(metadatos)}"
    faltan = set(f.ESPECIES) - set(metadatos["especie"])
    assert not faltan, f"Orinoquía: ya no aparecen las especies {sorted(faltan)}"
    seleccion = f.orinoquia_seleccion(metadatos)
    assert len(seleccion) == 1_876, f"Orinoquía: el subconjunto del curso tenía 1.876 fotos y ahora tiene {len(seleccion)}"
    foto = reducir(pedir(f.LILA + "public/" + seleccion["ruta_lila"].iloc[0]).content)
    assert foto[:2] == b"\xff\xd8", "Orinoquía: lo que llegó no es un JPEG"
    return f"Orinoquía (LILA): {len(metadatos)} anotaciones, subconjunto de 1.876 fotos, descarga de una foto correcta"


def probar():
    return [_eurosat(), _caldas(), _orinoquia()]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
