#!/usr/bin/env python3
"""
Revisa qué elimina el paso "mismo inicio de texto" de `consolidar`: ¿duplicados reales o plantilla?

  python revisar_duplicados.py

Para cada noticia aceptada calcula dos huellas del texto:
  A = caracteres 0-600     (la que usa consolidar)
  B = caracteres 300-900   (salta el arranque, donde suelen estar avisos y plantillas)
y compara:
  duplicado confirmado   A y B coinciden: el texto es el mismo (misma noticia con otra URL, sindicación)
  falso duplicado        A coincide pero B no: solo comparten el arranque (plantilla, cookies, avisos);
                         son noticias distintas que consolidar está borrando por error

Lee solo las columnas necesarias y procesa carpeta por carpeta (poca memoria).
Escribe salida/duplicados_grupos.csv con los grupos más grandes para revisarlos a mano.
"""
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ARTS = Path("salida/articulos")
COLS = ["idioma", "anio", "diario", "url", "texto", "aceptada"]


def huella(s: pd.Series, a: int, b: int) -> pd.Series:
    return (s.str[a:b].str.lower().str.replace(r"\W+", "", regex=True)
            .map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))


def leer(carpeta: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(carpeta, columns=COLS)
    except Exception:  # alguna parte ilegible: se saltan (estado.py las localiza)
        trozos = []
        for p in sorted(carpeta.glob("parte_*.parquet")):
            try:
                trozos.append(pd.read_parquet(p, columns=COLS))
            except Exception:
                pass
        return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame(columns=COLS)


carpetas = sorted(p for p in ARTS.glob("*_*") if p.is_dir())
if not carpetas:
    raise SystemExit("No hay descargas en salida/articulos/")
partes = []
for c in tqdm(carpetas, desc="Leyendo carpetas", unit="carpeta"):
    x = leer(c)
    x = x[x.aceptada.astype(bool)].copy()
    if x.empty:
        continue
    t = x.texto.fillna("")
    x["hA"], x["hB"] = huella(t, 0, 600), huella(t, 300, 900)
    x["inicio"] = t.str[:140].str.replace(r"\s+", " ", regex=True)
    partes.append(x[["idioma", "anio", "diario", "url", "hA", "hB", "inicio"]])
d = pd.concat(partes, ignore_index=True).drop_duplicates("url")
n = len(d)

dupA = d.duplicated("hA")
dupB = d.duplicated("hB")
conf, falso = int((dupA & dupB).sum()), int((dupA & ~dupB).sum())
print(f"\nNoticias aceptadas (URLs distintas): {n:,}")
print(f"Eliminadas por huella A (lo que hace consolidar): {int(dupA.sum()):,} ({dupA.mean():.1%})")
print(f"  duplicados confirmados (también iguales en B):   {conf:,}")
print(f"  falsos duplicados (solo comparten el arranque):   {falso:,}  <- se borran por error")

g = d.groupby("hA").agg(n=("url", "size"), diarios=("diario", "nunique"), diario=("diario", "first"),
                        inicio=("inicio", "first"), ejemplo1=("url", "first"), ejemplo2=("url", "last"))
g = g[g.n > 1].sort_values("n", ascending=False)
grandes = g[g.n >= 10]
print(f"\nGrupos con 2+ textos iguales en A: {len(g):,} | con 10+: {len(grandes):,} "
      f"(borran {int((grandes.n - 1).sum()):,}; casi seguro plantilla)")
g.head(500).to_csv("salida/duplicados_grupos.csv", index=True)

por_diario = pd.DataFrame({"aceptadas": d.groupby("diario").size(),
                           "eliminadas_A": d[dupA].groupby("diario").size(),
                           "falsos": d[dupA & ~dupB].groupby("diario").size()}).fillna(0).astype(int)
por_diario["%_eliminado"] = (100 * por_diario.eliminadas_A / por_diario.aceptadas).round(1)
print("\nPor diario (los que más pierden):")
print(por_diario.sort_values("%_eliminado", ascending=False).head(12).to_string())

print("\nLos 8 grupos más grandes (inicio del texto repetido):")
for r in g.head(8).itertuples():
    print(f"  {r.n:>5} textos | {r.diarios} diario(s), p. ej. {r.diario}\n         {r.inicio[:120]!r}")

print("\n== Qué hacer ==")
if n and falso > 0.2 * max(int(dupA.sum()), 1):
    print(f"  Más del 20% de lo eliminado son falsos duplicados: consolida con la huella que salta el arranque:\n"
          f"    python recolectar_gdelt.py consolidar --cap-estricto --huella-desde 300")
else:
    print("  La mayoría de lo eliminado son duplicados reales (mismo texto): consolidar está bien como está.")
print("  Revisa a mano salida/duplicados_grupos.csv (ejemplo1/ejemplo2 son URLs del mismo grupo).")