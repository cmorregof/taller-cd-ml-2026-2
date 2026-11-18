"""Genera los cuadernos de la clase 10 (incertidumbre, tablero, contar el resultado)."""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno
from clase10_a import CIFRA

destino = RAIZ / "clases" / "10_incertidumbre"
destino.mkdir(parents=True, exist_ok=True)
nbf.write(cuaderno(CIFRA, False), destino / "10a_la_cifra_que_falta.ipynb")
try:
    from clase10_b import VUELOS_ESTUDIANTE, VUELOS_SOLUCION
    nbf.write(cuaderno(VUELOS_ESTUDIANTE, False), destino / "10b_el_mapa_de_vuelos.ipynb")
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(VUELOS_SOLUCION, False), RAIZ / "_profesor" / "10b_el_mapa_de_vuelos_SOLUCION.ipynb")
except ImportError:
    print("Falta clase10_b.py: solo se escribió la demostración.")
print("Cuadernos escritos.")
