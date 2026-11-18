"""Genera los cuadernos de la clase 8 (señales y audio).

La demostración sale completa. El gemelo de los estudiantes lleva solo títulos, preguntas
y celdas vacías con un comentario; la solución completa va a `_profesor/`.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno
from clase08_a import ECG

destino = RAIZ / "clases" / "08_senales"
destino.mkdir(parents=True, exist_ok=True)
nbf.write(cuaderno(ECG, False), destino / "08a_el_electrocardiograma.ipynb")
try:
    from clase08_b import AVES_ESTUDIANTE, AVES_SOLUCION
    nbf.write(cuaderno(AVES_ESTUDIANTE, False), destino / "08b_los_sonidos_del_magdalena.ipynb")
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(AVES_SOLUCION, False), RAIZ / "_profesor" / "08b_los_sonidos_del_magdalena_SOLUCION.ipynb")
except ImportError:
    print("Falta clase08_b.py: solo se escribió la demostración.")
print("Cuadernos escritos.")
