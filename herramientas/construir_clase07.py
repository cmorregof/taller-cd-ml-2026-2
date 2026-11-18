"""Genera los cuadernos de la clase 7 (texto).

La demostración (Lo que dicen los turistas) sale completa. El gemelo (La plata
pública) usa el andamiaje de las clases 5 a 8: las celdas ("tarea", comentario,
solución) salen solo con el comentario en el cuaderno del estudiante.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import RAIZ, cuaderno
from construir_clase05 import gemelo
from clase07_turistas import TURISTAS
from clase07_plata import PLATA

if __name__ == "__main__":
    destino = RAIZ / "clases" / "07_texto"
    destino.mkdir(parents=True, exist_ok=True)
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(TURISTAS, False), destino / "07a_lo_que_dicen_los_turistas.ipynb")
    nbf.write(gemelo(PLATA, True), destino / "07b_la_plata_publica.ipynb")
    nbf.write(gemelo(PLATA, False), RAIZ / "_profesor" / "07b_la_plata_publica_SOLUCION.ipynb")
    print("Cuadernos de la clase 7 escritos.")
