"""Genera los cuadernos de la clase 6 (cambio de dominio).

La demostración (La tierra desde arriba) sale completa. El gemelo (Fauna de la
Orinoquía) usa el andamiaje de las clases 5 a 8: las celdas ("tarea", comentario,
solución) salen solo con el comentario en el cuaderno del estudiante.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import RAIZ, cuaderno
from construir_clase05 import gemelo
from clase06_tierra import TIERRA
from clase06_fauna import FAUNA

if __name__ == "__main__":
    destino = RAIZ / "clases" / "06_cambio_de_dominio"
    destino.mkdir(parents=True, exist_ok=True)
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(TIERRA, False), destino / "06a_la_tierra_desde_arriba.ipynb")
    nbf.write(gemelo(FAUNA, True), destino / "06b_fauna_de_la_orinoquia.ipynb")
    nbf.write(gemelo(FAUNA, False), RAIZ / "_profesor" / "06b_fauna_de_la_orinoquia_SOLUCION.ipynb")
    print("Cuadernos de la clase 6 escritos.")
