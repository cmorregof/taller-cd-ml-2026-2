"""Genera los cuadernos de la clase 9 (API, despliegue, agentes).

El servicio de la demostración vive en `clases/09_api/noticiero/` y la plantilla del gemelo
en `clases/09_api/parte_de_incendios/`: son archivos del repositorio, no se generan aquí.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno
from clase09_a import NOTICIERO

destino = RAIZ / "clases" / "09_api"
destino.mkdir(parents=True, exist_ok=True)
nbf.write(cuaderno(NOTICIERO, False), destino / "09a_el_noticiero_de_un_minuto.ipynb")
try:
    from clase09_b import PARTE_ESTUDIANTE, PARTE_SOLUCION
    nbf.write(cuaderno(PARTE_ESTUDIANTE, False), destino / "09b_el_parte_de_incendios.ipynb")
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(PARTE_SOLUCION, False), RAIZ / "_profesor" / "09b_el_parte_de_incendios_SOLUCION.ipynb")
except ImportError:
    print("Falta clase09_b.py: solo se escribió la demostración.")
print("Cuadernos escritos.")
