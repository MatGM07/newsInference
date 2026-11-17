#!/usr/bin/env python3
"""
Estado de la recolección: qué idioma-año está completo, cuál quedó a medias y cuál no ha empezado.
Solo lee (no descarga ni modifica nada), salvo con --mover-corruptos.

  python estado.py
  python estado.py --mover-corruptos    # aparta a salida/corruptos/ los .parquet ilegibles

Compara las aceptadas de salida/articulos/<idioma>_<año>/ con metas.csv:
  COMPLETO      aceptadas >= meta (descargar lo salta)
  PARCIAL       faltan aceptadas y aún hay margen de intentos: volver a lanzar descargar lo retoma
  LIMITE        faltan aceptadas pero los intentos ya llegaron a --max-intentos x meta:
                descargar NO sigue; para continuar hay que subir --max-intentos
  SIN_INICIAR   no hay carpeta o está vacía

Un .parquet ilegible suele ser un archivo que quedó a medias al interrumpir con Ctrl+C justo al
guardar. Hace fallar a `descargar` al arrancar: muévelo (--mover-corruptos); lo único que se
pierde es ese lote (<= 60 descargas), que se vuelve a intentar.
"""
import argparse
import math
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--metas", default="metas.csv")
ap.add_argument("--max-intentos", type=float, default=5.0, dest="max_intentos",
                help="El mismo valor que uses en descargar (por defecto 5)")
ap.add_argument("--tope", type=float, default=0.30)
ap.add_argument("--mover-corruptos", action="store_true", dest="mover")
args = ap.parse_args()

ARTS = Path("salida/articulos")
metas = pd.read_csv(args.metas).sort_values(["anio", "idioma"])
filas, corruptos = [], []

for m in metas.itertuples():
    carpeta = ARTS / f"{m.idioma}_{m.anio}"
    partes = sorted(carpeta.glob("parte_*.parquet")) if carpeta.exists() else []
    trozos = []
    for p in partes:
        try:
            trozos.append(pd.read_parquet(p, columns=["url", "diario", "aceptada"]))
        except Exception as e:
            corruptos.append((p, type(e).__name__))
    d = pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame(columns=["url", "diario", "aceptada"])
    meta = int(m.meta)
    acept = int(d.aceptada.sum()) if len(d) else 0
    intentos = d.url.nunique() if len(d) else 0
    por_diario = d[d.aceptada.astype(bool)].diario.value_counts() if acept else pd.Series(dtype=int)
    if not partes:
        estado = "SIN_INICIAR"
    elif acept >= meta:
        estado = "COMPLETO"
    elif intentos >= int(args.max_intentos * meta):
        estado = "LIMITE"
    else:
        estado = "PARCIAL"
    top = f"{por_diario.index[0]} {por_diario.iloc[0] / acept:.0%}" if acept else "-"
    filas.append({"idioma": m.idioma, "anio": m.anio, "meta": meta, "aceptadas": acept,
                  "%meta": round(100 * acept / meta) if meta else 0, "intentos": intentos,
                  "estado": estado, "diario_dominante": top})

res = pd.DataFrame(filas)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print("\nResumen:", res.estado.value_counts().to_dict(),
      f"| aceptadas {res.aceptadas.sum():,} de {res.meta.sum():,}")

extra = sorted(p.name for p in ARTS.glob("*_*") if p.is_dir()
               and p.name not in {f"{r.idioma}_{r.anio}" for r in res.itertuples()})
if extra:
    print("Carpetas sin meta en metas.csv:", ", ".join(extra))

if corruptos:
    print(f"\n!! {len(corruptos)} archivo(s) ilegible(s) (descargar fallará hasta que los muevas):")
    for p, e in corruptos:
        print(f"   {p}  [{e}]")
    if args.mover:
        dest = Path("salida/corruptos")
        dest.mkdir(parents=True, exist_ok=True)
        for p, _ in corruptos:
            shutil.move(str(p), str(dest / f"{p.parent.name}__{p.name}"))
        print(f"   Movidos a {dest}. Ya puedes volver a lanzar descargar.")
    else:
        print("   Corre de nuevo con --mover-corruptos para apartarlos.")

pend = res[res.estado.isin(["PARCIAL", "SIN_INICIAR"])]
if len(pend):
    print("\nPor retomar (descargar continúa donde quedó):")
    for idioma, g in pend.groupby("idioma"):
        print(f"  python recolectar_gdelt.py descargar --idiomas {idioma} "
              f"--anios {','.join(str(a) for a in sorted(g.anio.unique()))}")
lim = res[res.estado == "LIMITE"]
if len(lim):
    print("\nEn LIMITE (no avanzarán sin subir --max-intentos):",
          ", ".join(f"{r.idioma} {r.anio}" for r in lim.itertuples()))
dom = res[res.diario_dominante.str.contains("%") & (res.aceptadas > 0)]
dom = dom[dom.diario_dominante.str.extract(r"(\d+)%")[0].astype(int) > args.tope * 100]
if len(dom):
    print("Con un diario por encima del tope (se corrige con consolidar --cap-estricto):",
          ", ".join(f"{r.idioma} {r.anio} ({r.diario_dominante})" for r in dom.itertuples()))