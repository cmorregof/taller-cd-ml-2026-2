"""El parte de incendios: la API que ustedes construyen.

Lean README.md antes de escribir una línea. Los datos ya están resueltos en datos.py.
Para correrla:      uvicorn main:app --reload
Para examinarla:    python pruebas_aceptacion.py
"""
from fastapi import FastAPI

from datos import cargar_focos

app = FastAPI(title="El parte de incendios")

# Aquí va su API. Empiecen por GET /salud y corran las pruebas después de cada ruta.
