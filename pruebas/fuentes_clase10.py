"""Revisa las fuentes de la clase 10 con consultas pequeñas.

    python pruebas/fuentes_clase10.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests

from taller import fuentes_clase10


def probar():
    """Devuelve una lista de mensajes. Lanza AssertionError con un mensaje claro si una fuente cambió."""
    mensajes = []

    # DANE, catálogo 795: que el archivo de réplicas siga ahí y pese lo mismo (no se descarga completo)
    cabeza = requests.get(fuentes_clase10.RECLUTAMIENTO, stream=True, timeout=120, headers={"User-Agent": "taller-cd-ml-2026-2"})
    assert cabeza.status_code == 200, f"El DANE respondió {cabeza.status_code} para Reclutamiento.parquet.zip"
    primeros = next(cabeza.iter_content(4))
    peso = int(cabeza.headers.get("Content-Length", 0))
    cabeza.close()
    assert primeros[:2] == b"PK", "DANE 795: la descarga ya no es un archivo zip (¿apareció un captcha?)"
    assert peso in (0, 73498199), f"DANE 795: Reclutamiento.parquet.zip cambió de tamaño ({peso} bytes)"
    mensajes.append(f"DANE 795: Reclutamiento.parquet.zip disponible ({peso / 1e6:.1f} MB)")

    # Aerocivil: columnas y que La Nubia tenga datos
    filas = requests.get(fuentes_clase10.VUELOS, timeout=120, params={"$where": "origen='MZL' AND a_o=2022", "$limit": 5}).json()
    assert len(filas) > 0, "Aerocivil gb6w-ynu4: no hay trayectos con origen MZL en 2022"
    for columna in ["sigla_empresa", "nombre", "a_o", "n_mero_de_mes", "origen", "destino", "tipo_vuelo", "pasajeros"]:
        assert columna in filas[0], f"Aerocivil gb6w-ynu4: falta la columna {columna}"
    mensajes.append(f"Aerocivil gb6w-ynu4: responde, columnas {sorted(filas[0])[:4]}…")

    # OurAirports
    aeropuertos = fuentes_clase10.construir_aeropuertos()
    nubia = aeropuertos[aeropuertos["iata_code"] == "MZL"]
    assert len(nubia) == 1, "OurAirports: no aparece MZL"
    assert abs(float(nubia["latitude_deg"].iloc[0]) - 5.03) < 0.05, "OurAirports: las coordenadas de MZL cambiaron"
    mensajes.append(f"OurAirports: {len(aeropuertos)} aeropuertos con sigla IATA; MZL en ({float(nubia['latitude_deg'].iloc[0])}, {float(nubia['longitude_deg'].iloc[0])})")
    return mensajes


if __name__ == "__main__":
    for mensaje in probar():
        print("OK    ", mensaje)
