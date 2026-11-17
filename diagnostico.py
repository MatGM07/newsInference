#!/usr/bin/env python3
"""
Diagnóstico de lo ya descargado: qué diarios rinden y cuáles hacen perder tiempo.

  python diagnostico.py

Lee salida/articulos/*/parte_*.parquet (lo que lleve descargado `descargar`) y muestra,
por diario: intentos, % aceptadas, % sin descarga (bloqueos/muros de pago), % que
necesitó Wayback Machine y cuántas aceptadas salen por cada 100 intentos.
Los diarios con 'acept%' bajo o 'sin_desc%' alto son los que conviene pasar a
rol = reserva en diarios.csv (o cambiar por otro de la lista de reservas).
"""
from pathlib import Path

import pandas as pd

partes = sorted(Path("salida/articulos").glob("*/parte_*.parquet"))
if not partes:
    raise SystemExit("Todavía no hay descargas en salida/articulos/")
cols = ["idioma", "anio", "diario", "tipo", "rol", "via", "error", "aceptada"]
d = pd.concat((pd.read_parquet(p, columns=cols) for p in partes), ignore_index=True)

g = d.groupby(["idioma", "diario"])
t = pd.DataFrame({
    "tipo": g.tipo.first(),
    "intentos": g.size(),
    "aceptadas": g.aceptada.sum(),
    "acept%": (g.aceptada.mean() * 100).round(0),
    "sin_desc%": (g.error.apply(lambda s: (s == "sin_descarga").mean()) * 100).round(0),
    "no_ia%": (g.error.apply(lambda s: (s == "no_ia").mean()) * 100).round(0),
    "wayback%": (g.via.apply(lambda s: (s == "wayback").mean()) * 100).round(0),
}).sort_values(["idioma", "acept%"], ascending=[True, False])

pd.set_option("display.width", 200)
print(f"Intentos: {len(d):,} | aceptadas: {int(d.aceptada.sum()):,} "
      f"({d.aceptada.mean() * 100:.0f}%)\n")
print(t.to_string())
print("\nPor idioma y año (aceptadas / intentos):")
print(d.groupby(["idioma", "anio"]).aceptada.agg(["sum", "size"]).astype(int).to_string())
malos = t[(t["sin_desc%"] >= 50) | (t["acept%"] < 15)]
if len(malos):
    print("\nCandidatos a pasar a reserva (sin_desc% >= 50 o acept% < 15):")
    print(", ".join(malos.index.get_level_values("diario")))