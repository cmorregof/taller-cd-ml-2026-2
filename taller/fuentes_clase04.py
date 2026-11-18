"""Datos de la clase 4: los remates del Mundial de Qatar y los siniestros viales de Bogotá.

- Mundial. StatsBomb publica en GitHub el evento a evento de los 64 partidos. Su
  licencia no permite redistribuir, así que cada quien lo lee de GitHub y el
  curso no guarda copia de respaldo.
- Bogotá. La Secretaría Distrital de Movilidad publica el anuario de siniestralidad
  como un Excel de 127 MB. Se baja, se lee la hoja «Siniestros» y se suelta.
"""
from __future__ import annotations

import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from . import clase04_xlsx

# ------------------------------------------------------------------ Mundial
STATSBOMB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETICION, TEMPORADA = 43, 106          # Copa Mundial masculina de la FIFA, 2022
ARCO_X, ARCO_Y, MEDIO_ARCO = 120.0, 40.0, 4.0      # la cancha de StatsBomb mide 120 por 80 y los palos están en y = 36 y y = 44


def _json(url: str):
    ultimo = None
    for intento in range(4):
        try:
            respuesta = requests.get(url, timeout=120)
            respuesta.raise_for_status()
            return respuesta.json()
        except Exception as error:  # se reintenta con espera creciente
            ultimo = error
            time.sleep(2 * (intento + 1))
    raise RuntimeError(f"GitHub no respondió para {url}") from ultimo


def partidos(competicion: int = COMPETICION, temporada: int = TEMPORADA) -> pd.DataFrame:
    """Los partidos de un torneo, en orden de fecha. Por defecto, los 64 del Mundial de 2022."""
    filas = [{"partido": p["match_id"], "fecha": p["match_date"], "hora": p["kick_off"],
              "fase": p["competition_stage"]["name"],
              "local": p["home_team"]["home_team_name"], "visitante": p["away_team"]["away_team_name"],
              "goles_local": p["home_score"], "goles_visitante": p["away_score"]}
             for p in _json(f"{STATSBOMB}/matches/{competicion}/{temporada}.json")]
    return pd.DataFrame(filas).sort_values(["fecha", "hora", "partido"]).reset_index(drop=True)


def _en_el_cono(x: float, y: float, px: float, py: float) -> bool:
    """¿El punto (px, py) está dentro del triángulo que forman el remate y los dos palos?"""
    def lado(ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d1 = lado(x, y, ARCO_X, ARCO_Y - MEDIO_ARCO)
    d2 = lado(ARCO_X, ARCO_Y - MEDIO_ARCO, ARCO_X, ARCO_Y + MEDIO_ARCO)
    d3 = lado(ARCO_X, ARCO_Y + MEDIO_ARCO, x, y)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def remates_de(partido: int) -> list[dict]:
    """Los remates de un partido, uno por fila, con lo que StatsBomb anota de cada uno."""
    filas = []
    for evento in _json(f"{STATSBOMB}/events/{partido}.json"):
        if evento["type"]["name"] != "Shot":
            continue
        remate = evento["shot"]
        x, y = evento["location"][:2]
        rivales = [j for j in remate.get("freeze_frame", []) if not j["teammate"]]
        arquero = next((j for j in rivales if j["position"]["name"] == "Goalkeeper"), None)
        filas.append({
            "partido": partido, "periodo": evento["period"], "minuto": evento["minute"], "segundo": evento["second"],
            "equipo": evento["team"]["name"], "jugador": evento["player"]["name"],
            "x": x, "y": y,
            "tipo": remate["type"]["name"], "parte_del_cuerpo": remate["body_part"]["name"],
            "tecnica": remate["technique"]["name"], "jugada": evento["play_pattern"]["name"],
            "bajo_presion": int(evento.get("under_pressure", False)),
            "de_primera": int(remate.get("first_time", False)),
            "mano_a_mano": int(remate.get("one_on_one", False)),
            "rivales_en_el_cono": sum(_en_el_cono(x, y, *j["location"]) for j in rivales if j is not arquero) if "freeze_frame" in remate else None,
            "arquero_en_el_cono": (int(_en_el_cono(x, y, *arquero["location"])) if arquero else 0) if "freeze_frame" in remate else None,
            "distancia_del_arquero_al_arco": (math.hypot(ARCO_X - arquero["location"][0], ARCO_Y - arquero["location"][1]) if arquero else None),
            "resultado": remate["outcome"]["name"], "gol": int(remate["outcome"]["name"] == "Goal"),
            "statsbomb_xg": remate["statsbomb_xg"],
        })
    return filas


def construir_remates(competicion: int = COMPETICION, temporada: int = TEMPORADA) -> pd.DataFrame:
    """Todos los remates de un torneo, incluidas las tandas de penales (periodo 5).

    Por defecto es el Mundial de 2022 (43, 106). Otros torneos abiertos de StatsBomb:
    Copa América 2024 (223, 282), Mundial femenino 2023 (72, 107), Eurocopa 2024 (55, 282).
    """
    calendario = partidos(competicion, temporada)
    print(f"   StatsBomb, {len(calendario)} partidos…", flush=True)
    with ThreadPoolExecutor(max_workers=8) as grupo:
        por_partido = list(grupo.map(remates_de, calendario["partido"]))
    remates = pd.DataFrame([fila for filas in por_partido for fila in filas])
    remates = remates.merge(calendario[["partido", "fecha", "fase", "local", "visitante"]], on="partido")
    return remates.sort_values(["fecha", "partido", "periodo", "minuto", "segundo"]).reset_index(drop=True)


# ------------------------------------------------------------------ Bogotá
ANUARIO = ("https://observatorio.movilidadbogota.gov.co/sites/default/files/2025-09/"
           "Base_Anuario_de_Siniestralidad_2024.xlsx")
# De la hoja Siniestros se dejan por fuera el número de formulario y las diez
# columnas que detallan el tipo de bus (Con_Sitp, Con_Troncal, Con_Zonal…).
COLUMNAS_SINIESTROS = [
    "Codigo_Accidente", "Longitud", "Latitud", "Direccion", "Fecha_Acc", "AA_Acc", "MM_Acc", "Dia_Semana_Acc",
    "Hora_Acc", "Localidad", "Clase_Acc", "Elemento_Choque", "Gravedad_Indicador_Tradicional",
    "Gravedad_indicador_30d", "Con_Bicicleta", "Con_Carga", "Con_Embriaguez", "Con_Huecos", "Con_Menores",
    "Con_Moto", "Con_Peaton", "Con_Persona_Mayor", "Con_Tpi", "Con_Tpp", "Con_Velocidad",
]


def bajar_anuario(destino: Path) -> Path:
    """Baja el Excel oficial a `destino`, por pedazos, sin cargarlo en memoria."""
    with requests.get(ANUARIO, stream=True, timeout=300) as respuesta:
        respuesta.raise_for_status()
        with open(destino, "wb") as archivo:
            for pedazo in respuesta.iter_content(chunk_size=1 << 20):
                archivo.write(pedazo)
    return destino


def construir_siniestros() -> pd.DataFrame:
    """Siniestros viales de Bogotá, 2015 a 2024: una fila por siniestro, como los publica la fuente."""
    with tempfile.TemporaryDirectory() as carpeta:
        print("   Secretaría de Movilidad, anuario de 127 MB…", flush=True)
        ruta = bajar_anuario(Path(carpeta) / "anuario.xlsx")
        print("   Leo la hoja Siniestros…", flush=True)
        tabla = clase04_xlsx.leer_hoja(ruta, "Siniestros", columnas=COLUMNAS_SINIESTROS)
    if len(tabla) < 250_000:
        raise RuntimeError(f"La hoja Siniestros trae {len(tabla)} filas y se esperaban unas 278.000.")
    return tabla


FICHAS = {
    "remates_mundial": {
        "titulo": "Remates de un torneo de fútbol (por defecto, el Mundial de Qatar 2022), con posición, contexto y resultado",
        "fuente": "StatsBomb Open Data, en GitHub",
        "url": "https://github.com/statsbomb/open-data",
        "licencia": "Uso con cita a StatsBomb. La licencia no permite redistribuir: se lee de GitHub y no hay copia de respaldo.",
        "constructor": construir_remates,
        "md5": None,
    },
    "siniestros_bogota": {
        "titulo": "Bogotá: siniestros viales, 2015 a 2024",
        "fuente": "Secretaría Distrital de Movilidad de Bogotá, Base del Anuario de Siniestralidad 2024, hoja Siniestros",
        "url": ANUARIO,
        "licencia": "Datos abiertos de la Secretaría Distrital de Movilidad. Citar la fuente.",
        "constructor": construir_siniestros,
        "md5": None,
    },
}
