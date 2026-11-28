"""Datos de la clase 6: EuroSAT, parches de Sentinel-2 de Caldas y cámaras trampa de la Orinoquía.

Los tres son conjuntos de imágenes y siguen el esquema de la clase 5: `descargar`
entrega la tabla y las fotos quedan en `datos/cache/<carpeta>/`.
"""
from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

from .clase05_imagenes import CACHE, en_paralelo, pedir, reducir

# ------------------------------------------------------------------- EuroSAT
EUROSAT_ZIP = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"
EUROSAT_POR_CLASE = 400


def construir_eurosat() -> pd.DataFrame:
    """Los parches numerados de 1 a 400 de cada una de las diez clases de EuroSAT RGB.

    El ZIP completo (94 MB, 27.000 parches de 64 x 64 píxeles) se baja entero y de él
    se guardan 4.000 parches, sin modificarlos.
    """
    from PIL import Image

    archivo = zipfile.ZipFile(io.BytesIO(pedir(EUROSAT_ZIP, timeout=600).content))
    carpeta = CACHE / "eurosat"
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for nombre in archivo.namelist():
        coincide = re.search(r"/([A-Za-z]+)_(\d+)\.jpg$", nombre)
        if coincide and int(coincide.group(2)) <= EUROSAT_POR_CLASE:
            clase, numero = coincide.group(1), int(coincide.group(2))
            (carpeta / f"{clase}_{numero}.jpg").write_bytes(archivo.read(nombre))
            filas.append({"archivo": f"{clase}_{numero}.jpg", "clase": clase, "numero": numero})
    tabla = pd.DataFrame(filas).sort_values(["clase", "numero"]).reset_index(drop=True)
    conteo = tabla["clase"].value_counts()
    if len(conteo) != 10 or (conteo != EUROSAT_POR_CLASE).any():
        raise ValueError(f"EuroSAT cambió: esperaba 10 clases con {EUROSAT_POR_CLASE} parches y hay {conteo.to_dict()}")
    if Image.open(carpeta / tabla["archivo"][0]).size != (64, 64):
        raise ValueError("EuroSAT cambió: los parches ya no son de 64 x 64 píxeles")
    return tabla


# -------------------------------------------------------------------- Caldas
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_DATOS = "https://planetarycomputer.microsoft.com/api/data/v1/item"
ESCENA = "S2B_MSIL2A_20260810T152649_R025_T18NVL_20260810T204206"   # 10 de agosto de 2026, 3,7 % de nubes
WORLDCOVER = "ESA_WorldCover_10m_2021_v200_N03W078"
# Esquina superior izquierda de cada zona, en metros UTM 18N (EPSG:32618), alineada a la malla de 10 m de la tesela T18NVL.
ZONAS = {
    "cafetera": (424000, 572000),   # Manizales, Villamaría, Chinchiná, Palestina, Neira
    "nevados": (452000, 553600),    # Nevado del Ruiz y el páramo que lo rodea
}
LADO_ZONA = 2560          # píxeles de 10 m: 25,6 km de lado
PARCHE = 64               # el tamaño de los parches de EuroSAT: 640 m
# Leyenda de ESA WorldCover, tomada de los metadatos del propio producto (classification:classes).
LEYENDA = {10: "arbolado", 20: "arbustos", 30: "pastizal", 40: "cultivo", 50: "construido",
           60: "suelo desnudo", 70: "nieve y hielo", 80: "agua", 90: "humedal", 95: "manglar", 100: "musgo y liquen"}
COLORES = {10: "#006400", 20: "#FFBB22", 30: "#FFFF4C", 40: "#F096FF", 50: "#FA0000",
           60: "#B4B4B4", 70: "#F0F0F0", 80: "#0064C8", 90: "#0096A0", 95: "#00CF75", 100: "#FAE6A0"}
NUBES_SCL = [3, 8, 9, 10]   # clasificación de escena de Sentinel-2: sombra de nube, nube media, nube alta, cirro


def recorte_pc(coleccion: str, item: str, banda: str, x0: int, y1: int, n: int, vecino: bool = False) -> np.ndarray:
    """Pide a Planetary Computer un recorte de n x n píxeles de 10 m en UTM 18N, sin remuestrear."""
    caja = f"{x0},{y1 - n * 10},{x0 + n * 10},{y1}"
    parametros = {"collection": coleccion, "item": item, "assets": banda,
                  "coord_crs": "epsg:32618", "dst_crs": "epsg:32618"}
    if vecino:
        parametros.update({"resampling": "nearest", "reproject": "nearest"})
    respuesta = pedir(f"{PC_DATOS}/bbox/{caja}/{n}x{n}.npy", params=parametros, timeout=180)
    return np.load(io.BytesIO(respuesta.content))


def _mosaico(coleccion, item, banda, x0, y1, capas, vecino):
    mitad = LADO_ZONA // 2
    lienzo = np.zeros((LADO_ZONA, LADO_ZONA, capas), np.uint8)
    for i in range(2):
        for j in range(2):
            bloque = recorte_pc(coleccion, item, banda, x0 + j * mitad * 10, y1 - i * mitad * 10, mitad, vecino)
            lienzo[i * mitad:(i + 1) * mitad, j * mitad:(j + 1) * mitad] = bloque[:capas].transpose(1, 2, 0)
    return lienzo


def construir_caldas(zona: str = "cafetera") -> pd.DataFrame:
    """1.600 parches reales de Sentinel-2 (640 m de lado) de una zona de Caldas, con una referencia de cobertura.

    Tres capas de la misma zona, todas de Planetary Computer:
    - el color verdadero (`visual`) de la escena de Sentinel-2 L2A del 10 de agosto de 2026;
    - su clasificación de escena (`SCL`), para saber qué parches tienen nubes;
    - ESA WorldCover 2021, que da la cobertura de referencia de cada píxel.
    La referencia de un parche es la cobertura más frecuente entre sus 4.096 píxeles.
    """
    from PIL import Image

    if zona not in ZONAS:
        raise KeyError(f"No conozco la zona «{zona}». Las disponibles son: {', '.join(ZONAS)}.")
    x0, y1 = ZONAS[zona]
    print(f"   Planetary Computer: color, nubes y WorldCover de la zona {zona}…", flush=True)
    color = _mosaico("sentinel-2-l2a", ESCENA, "visual", x0, y1, 3, False)
    escena = _mosaico("sentinel-2-l2a", ESCENA, "SCL", x0, y1, 1, True)[:, :, 0]
    cobertura = _mosaico("esa-worldcover", WORLDCOVER, "map", x0, y1, 1, True)[:, :, 0]
    if color.max() == 0 or not set(np.unique(cobertura)) <= set(LEYENDA) | {0}:
        raise ValueError("Planetary Computer devolvió un recorte vacío o una leyenda de WorldCover desconocida")

    carpeta = CACHE / f"caldas_{zona}"
    carpeta.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(carpeta / "mosaico.jpg", quality=92)
    paleta = np.zeros((256, 3), np.uint8)
    for codigo, tono in COLORES.items():
        paleta[codigo] = [int(tono[k:k + 2], 16) for k in (1, 3, 5)]
    Image.fromarray(paleta[cobertura]).save(carpeta / "referencia.png")

    filas = []
    for i in range(LADO_ZONA // PARCHE):
        for j in range(LADO_ZONA // PARCHE):
            corte = (slice(i * PARCHE, (i + 1) * PARCHE), slice(j * PARCHE, (j + 1) * PARCHE))
            nombre = f"f{i:02d}_c{j:02d}.png"
            Image.fromarray(color[corte]).save(carpeta / nombre)
            codigos, cuentas = np.unique(cobertura[corte], return_counts=True)
            fracciones = dict(zip(codigos.tolist(), (cuentas / PARCHE ** 2).tolist()))
            mayor = int(codigos[cuentas.argmax()])
            fila = {"archivo": nombre, "zona": zona, "fila": i, "columna": j,
                    "x_utm": x0 + j * PARCHE * 10 + PARCHE * 5, "y_utm": y1 - i * PARCHE * 10 - PARCHE * 5,
                    "referencia": LEYENDA.get(mayor, "sin dato"), "pureza": round(float(cuentas.max() / PARCHE ** 2), 4),
                    "nubes": round(float(np.isin(escena[corte], NUBES_SCL).mean()), 4)}
            for codigo in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                fila[f"frac_{LEYENDA[codigo].replace(' ', '_')}"] = round(fracciones.get(codigo, 0.0), 4)
            filas.append(fila)
    return pd.DataFrame(filas)


# ----------------------------------------------------------------- Orinoquía
LILA = "https://storage.googleapis.com/public-datasets-lila/orinoquia-camera-traps/"
ESPECIES = ["collared_peccary", "black_agouti", "spotted_paca", "south_american_coati",
            "lowland_tapir", "southern_tamandua", "giant_anteater", "spixs_guan"]
SECUENCIAS_POR_ESPECIE = 250


def orinoquia_metadatos() -> pd.DataFrame:
    """Una fila por imagen anotada, a partir del archivo COCO que publica LILA."""
    paquete = zipfile.ZipFile(io.BytesIO(pedir(LILA + "orinoquia_camera_traps_metadata.zip").content))
    coco = json.loads(paquete.read(paquete.namelist()[0]))
    nombres = {c["id"]: c["name"] for c in coco["categories"]}
    anotaciones = pd.DataFrame(coco["annotations"])
    anotaciones["especie"] = anotaciones["category_id"].map(nombres)
    imagenes = pd.DataFrame(coco["images"])
    tabla = imagenes.merge(anotaciones[["image_id", "especie"]], left_on="id", right_on="image_id")
    return tabla[["file_name", "location", "seq_id", "frame_num", "seq_num_frames", "datetime", "especie"]]


def orinoquia_seleccion(metadatos: pd.DataFrame) -> pd.DataFrame:
    """El subconjunto del curso: ocho especies, una sola foto por secuencia, hasta 250 secuencias por especie.

    De cada secuencia se toma el cuadro central entre los que tienen la especie. Cuando
    una especie tiene más de 250 secuencias, se ordenan por cámara y fecha y se toman
    250 repartidas de manera uniforme. No hay sorteo: la selección es siempre la misma.
    """
    partes = []
    for especie in ESPECIES:
        de_la_especie = metadatos[metadatos["especie"] == especie].sort_values(["seq_id", "frame_num", "file_name"])
        centro = de_la_especie.groupby("seq_id", sort=False).apply(lambda g: g.iloc[len(g) // 2], include_groups=False).reset_index()
        centro = centro.sort_values(["location", "datetime", "seq_id"]).reset_index(drop=True)
        if len(centro) > SECUENCIAS_POR_ESPECIE:
            centro = centro.iloc[np.linspace(0, len(centro) - 1, SECUENCIAS_POR_ESPECIE).astype(int)]
        partes.append(centro.assign(especie=especie))
    seleccion = pd.concat(partes, ignore_index=True).drop_duplicates("file_name")
    seleccion["archivo"] = seleccion["file_name"].str.replace("/", "_")
    seleccion = seleccion.rename(columns={"location": "camara", "datetime": "fecha", "file_name": "ruta_lila"})
    return seleccion[["archivo", "especie", "camara", "seq_id", "frame_num", "seq_num_frames", "fecha", "ruta_lila"]]


def construir_orinoquia() -> pd.DataFrame:
    """Cerca de 1.900 fotos de cámaras trampa de la Orinoquía colombiana, reducidas a 256 píxeles de lado corto."""
    tabla = orinoquia_seleccion(orinoquia_metadatos())
    carpeta = CACHE / "orinoquia"
    carpeta.mkdir(parents=True, exist_ok=True)

    def bajar(fila):
        destino = carpeta / fila["archivo"]
        if not destino.exists():
            destino.write_bytes(reducir(pedir(LILA + "public/" + fila["ruta_lila"]).content))

    en_paralelo(bajar, tabla.to_dict("records"), hilos=16, aviso="fotos de la Orinoquía", cada=250)
    return tabla.sort_values(["camara", "fecha", "archivo"]).reset_index(drop=True)


FICHAS = {
    "eurosat": {
        "titulo": "EuroSAT RGB: parches de Sentinel-2 de 64 x 64 píxeles en diez clases de cobertura (400 por clase)",
        "fuente": "Helber, Bischke, Dengel y Borth (2019), copia de torchgeo en Hugging Face",
        "url": "https://huggingface.co/datasets/torchgeo/eurosat",
        "documentacion": "https://arxiv.org/abs/1709.00029",   # el artículo de los autores
        "licencia": "MIT. Citar a los autores.",
        "constructor": construir_eurosat,
        "md5": "b28e0463b261f2226b82890db3764fdd",
    },
    "caldas": {
        "titulo": "Caldas desde Sentinel-2: 1.600 parches de 640 m con referencia de ESA WorldCover 2021",
        "fuente": "Microsoft Planetary Computer: Sentinel-2 L2A (escena del 10 de agosto de 2026, tesela T18NVL) y ESA WorldCover 2021 v200",
        "url": "https://planetarycomputer.microsoft.com",
        "documentacion": ["https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",      # la página de cada colección
                          "https://planetarycomputer.microsoft.com/dataset/esa-worldcover"],
        "licencia": "Sentinel-2: datos Copernicus, uso libre con cita. ESA WorldCover: CC BY 4.0.",
        "constructor": construir_caldas,
        "md5": None,   # hay una tabla por zona: el MD5 se registra y no se compara
    },
    "orinoquia": {
        "titulo": "Orinoquía Camera Traps: ocho especies, una foto por secuencia",
        "fuente": "LILA BC, Orinoquía Camera Traps (Vélez et al., 2023)",
        "url": "https://lila.science/datasets/orinoquia-camera-traps/",
        "documentacion": "https://doi.org/10.1111/2041-210X.14044",   # el artículo de Vélez y colaboradores
        "licencia": "Community Data License Agreement (variante permisiva). Citar a los autores.",
        "constructor": construir_orinoquia,
        "md5": "ca3e6adb7e07778e427bb1aa3ac55235",
    },
}
