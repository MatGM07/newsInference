#!/usr/bin/env python3
"""
Valida el dataset consolidado y arma la carpeta entrega/ para quien siga con el modelado
(sentimiento / emociones / temas).

  python recolectar_gdelt.py consolidar --cap-estricto      # antes: genera salida/dataset_noticias.parquet
  python preparar_entrega.py                                 # después: valida y empaqueta

Genera en entrega/:
  noticias_<idioma>.parquet   el dataset, un archivo por idioma (en, es, pt, fr, de)
  LEEME_dataset.md            ficha del dataset: columnas, cobertura, limitaciones y siguientes pasos
  cobertura_idioma_anio.csv   final vs meta por idioma y año
  composicion_diarios.csv     aporte de cada diario por idioma y año
  serie_mensual.csv           noticias por idioma y mes, y % de diarios de tecnología
  muestra_revision.csv        300 noticias al azar (texto recortado) para revisar a mano
  validacion.txt              resultado de las comprobaciones
"""
import argparse
import numbers
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="salida/dataset_noticias.parquet")
ap.add_argument("--metas", default="metas.csv")
ap.add_argument("--tope", type=float, default=0.30)
ap.add_argument("--salida", default="entrega")
args = ap.parse_args()

d = pd.read_parquet(args.dataset)
metas = pd.read_csv(args.metas)
out = Path(args.salida)
out.mkdir(exist_ok=True)
avisos, info = [], []


def md_tabla(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    filas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        filas.append("| " + " | ".join(f"{v:,}" if isinstance(v, numbers.Integral) and not isinstance(v, bool) else str(v) for v in r) + " |")
    return "\n".join(filas)


# ------------------------------------------------------------------ comprobaciones
esperadas = ["id", "idioma", "diario", "tipo", "anio_gdelt", "fecha_final", "anio_pub", "mes_pub",
             "fecha_origen", "titulo", "texto", "n_palabras", "n_menciones_ia", "origen_url", "url"]
faltan = [c for c in esperadas if c not in d.columns]
if faltan:
    raise SystemExit(f"Al dataset le faltan columnas: {faltan}")

n = len(d)
info.append(f"Noticias: {n:,} | idiomas: {d.idioma.nunique()} | diarios: {d.diario.nunique()}")
for col in ("id", "url"):
    k = int(d[col].duplicated().sum())
    (avisos if k else info).append(f"Duplicados en {col}: {k}")
hay_titulo = d.titulo.fillna("").str.strip() != ""
pct_titulo = hay_titulo.mean() * 100
k = int((d.duplicated(["diario", "titulo"]) & hay_titulo).sum())
(avisos if k else info).append(f"Mismo diario y título (no vacío) repetidos: {k}")
if pct_titulo < 90:
    avisos.append(f"Títulos disponibles: solo {pct_titulo:.0f}% (la extracción no los guardó); la columna `titulo` queda mayormente vacía")
else:
    info.append(f"Títulos disponibles: {pct_titulo:.0f}%")
for col in ("texto", "fecha_final"):
    k = int(d[col].isna().sum() + (d[col].astype(str).str.strip() == "").sum()) if col != "fecha_final" else int(d[col].isna().sum())
    (avisos if k else info).append(f"Vacíos en {col}: {k}")

fuera = int((~d.fecha_final.between("2021-01-01", "2026-10-31")).sum())
(avisos if fuera else info).append(f"Fechas fuera de 2021-01 a 2026-10: {fuera}")
pct_gdelt = (d.fecha_origen == "gdelt").mean() * 100
info.append(f"Fecha tomada de la página: {100 - pct_gdelt:.0f}% | de GDELT (la página no traía fecha válida): {pct_gdelt:.0f}%")
desfase = (d.anio_pub != d.anio_gdelt).mean() * 100
info.append(f"Año de publicación distinto del año de GDELT: {desfase:.1f}%")
q = d.n_palabras.quantile([0.5, 0.95]).astype(int)
info.append(f"Palabras por texto: mediana {q[0.5]:,} | p95 {q[0.95]:,} | máx {int(d.n_palabras.max()):,}")

rep = d.texto.str.slice(0, 80).value_counts()
rep = rep[rep >= 5]
if len(rep):
    avisos.append(f"{len(rep)} inicios de texto se repiten >= 5 veces (posible texto de plantilla/cookies). "
                  f"El más repetido ({int(rep.iloc[0])} veces): {rep.index[0]!r}")

# cobertura final vs meta
fin = d.groupby(["idioma", "anio_gdelt"]).size().rename("final").reset_index().rename(columns={"anio_gdelt": "anio"})
cob = metas.merge(fin, on=["idioma", "anio"], how="left").fillna({"final": 0})
cob["final"] = cob.final.astype(int)
cob["pct_meta"] = (100 * cob.final / cob.meta).round(0).astype(int)
cob.sort_values(["anio", "idioma"]).to_csv(out / "cobertura_idioma_anio.csv", index=False)
bajos = cob[cob.pct_meta < 80]
if len(bajos):
    avisos.append("Idioma-año por debajo del 80% de su meta: "
                  + ", ".join(f"{r.idioma} {r.anio} ({r.pct_meta}%)" for r in bajos.itertuples()))
info.append(f"Total final: {n:,} de {int(metas.meta.sum()):,} de la meta ({100 * n / metas.meta.sum():.0f}%)")

# composición por diario
comp = (d.groupby(["idioma", "anio_gdelt", "diario", "tipo"]).size().rename("n").reset_index())
comp["pct_idioma_anio"] = (100 * comp.n / comp.groupby(["idioma", "anio_gdelt"]).n.transform("sum")).round(1)
comp.to_csv(out / "composicion_diarios.csv", index=False)
altos = comp[comp.pct_idioma_anio > args.tope * 100 + 0.5]
pct_max = comp.groupby(["idioma", "anio_gdelt"]).pct_idioma_anio.max()
if len(altos):
    avisos.append(f"{len(altos)} combinaciones diario-idioma-año pesan más del {args.tope:.0%} de su idioma-año "
                  f"(el tope se aplicó sobre la meta, no sobre lo entregado; máximo {pct_max.max():.0f}%). "
                  "Ejemplos: " + ", ".join(f"{r.diario} {r.idioma} {r.anio_gdelt} ({r.pct_idioma_anio:.0f}%)"
                                           for r in altos.sort_values("pct_idioma_anio", ascending=False).head(5).itertuples()))
tech = d.assign(es_tech=d.tipo == "tecnologia").groupby(["idioma", "anio_gdelt"]).es_tech.mean().mul(100).round(0).unstack()

# serie mensual
d["ym"] = d.fecha_final.dt.to_period("M").astype(str)
serie = (d.groupby(["idioma", "ym"]).agg(n=("id", "size"), pct_tecnologia=("tipo", lambda s: round(100 * (s == "tecnologia").mean(), 1)))
         .reset_index())
serie.to_csv(out / "serie_mensual.csv", index=False)

# muestra para revisión manual
m = d.sample(min(300, n), random_state=7)[["id", "idioma", "diario", "fecha_final", "titulo", "texto", "url"]].copy()
m["texto"] = m.texto.str.slice(0, 400)
m.to_csv(out / "muestra_revision.csv", index=False)

for idioma, g in d.drop(columns="ym").groupby("idioma"):
    g.to_parquet(out / f"noticias_{idioma}.parquet", index=False)

# ------------------------------------------------------------------ ficha del dataset
por_idioma = d.groupby("idioma").agg(noticias=("id", "size"), diarios=("diario", "nunique"),
                                      palabras_mediana=("n_palabras", "median")).reset_index()
por_idioma["palabras_mediana"] = por_idioma.palabras_mediana.astype(int)
por_anio = d.pivot_table(index="idioma", columns="anio_gdelt", values="id", aggfunc="count", fill_value=0)
por_anio.columns = [str(c) for c in por_anio.columns]
por_anio = por_anio.reset_index()
tech.columns = [str(c) for c in tech.columns]

nota_fecha = nota_titulo = ""
if pct_gdelt > 90:
    nota_fecha = ("7. **Fechas:** `fecha_final` viene de GDELT (fecha en que GDELT registró la noticia, normalmente muy cerca de la publicación), "
                  "no de la página. Es adecuada para agrupar por mes o trimestre, pero no para análisis al día.\n")
if pct_titulo < 90:
    nota_titulo = (f"{'8' if nota_fecha else '7'}. **Títulos:** solo {pct_titulo:.0f}% de las filas tienen `titulo`. No se podrá analizar el tono de los titulares "
                   "salvo que se recuperen aparte (volviendo a visitar las URLs de una muestra).\n")

leeme = f"""# Dataset: noticias sobre inteligencia artificial, 2021-2026

Corpus de **{n:,} noticias** sobre IA publicadas entre 2021 y octubre de 2026 en {d.diario.nunique()} diarios de 5 idiomas
(es, en, pt, fr, de). Objetivo del proyecto: ver cómo cambian los temas y el tono (sentimiento y emociones) con que se
escribe sobre la IA a lo largo del tiempo. Este paquete está listo para el paso de **modelado de sentimiento y emociones**.

## Archivos
- `noticias_<idioma>.parquet`: el dataset, un archivo por idioma. Usa estos.
- `cobertura_idioma_anio.csv`: noticias finales frente a la meta por idioma y año.
- `composicion_diarios.csv`: aporte de cada diario (y su tipo) por idioma y año.
- `serie_mensual.csv`: noticias por idioma y mes, y % de diarios de tecnología.
- `muestra_revision.csv`: 300 noticias al azar con texto recortado, para revisar a ojo.
- `validacion.txt`: resultado de las comprobaciones automáticas.

## Columnas
| columna | significado |
|---|---|
| `id` | identificador único (hash de la URL) |
| `idioma` | es, en, pt, fr, de (idioma del diario, confirmado con detección automática sobre el texto) |
| `diario`, `tipo` | medio y su tipo: `generalista` o `tecnologia` |
| `rol` | `principal` o `reserva` (diario de respaldo usado solo si faltaban noticias) |
| `fecha_final` | fecha de publicación: la de la página si es válida, si no la que reporta GDELT |
| `fecha_origen` | `pagina` o `gdelt`: de dónde salió `fecha_final` |
| `anio_pub`, `mes_pub` | año y mes de `fecha_final` |
| `anio_gdelt` | año con el que se muestreó la noticia (puede diferir de `anio_pub` en {desfase:.1f}% de los casos) |
| `titulo` | título de la noticia. **Disponible en {pct_titulo:.0f}% de las filas**; si está vacío, no se guardó al descargar |
| `texto` | cuerpo de la noticia extraído de la página |
| `n_palabras` | palabras del texto (mínimo 150) |
| `n_menciones_ia` | veces que el texto menciona términos de IA (ChatGPT, OpenAI, inteligencia artificial, KI, IA...) |
| `origen_url` | `match` (la URL traía una palabra clave de IA), `tech_sin_match` o `control` (la URL no la traía) |
| `via` | `directo` o `wayback` (descargada de Wayback Machine) |
| `url` | enlace original |

## Cobertura
{md_tabla(por_idioma)}

Noticias por idioma y año:

{md_tabla(por_anio)}

% de noticias de diarios de tecnología por idioma y año:

{md_tabla(tech.reset_index().astype(object).fillna("-"))}

## Limitaciones que hay que tener presentes
1. **La cobertura de IA crece muchísimo desde 2023** (efecto ChatGPT) y, además, GDELT indexa menos diarios en 2021-2022. Por eso 2021-2022
   tienen muchas menos noticias y varios idiomas-año quedaron bajo su meta. Analiza **proporciones** (por ejemplo % negativo por trimestre),
   no conteos, y agrupa 2021-2022 en periodos más largos.
2. **La composición de fuentes cambia con el tiempo** (ver `composicion_diarios.csv` y el % de tecnología de arriba). Un cambio de tono puede venir
   de un cambio de fuente y no del discurso. Incluye `diario` o `tipo` como variable de control o compara dentro del mismo tipo de diario.
3. **Tope por diario:** se limitó cada diario al {args.tope:.0%} de la *meta* de su idioma-año, no de lo que finalmente se entregó. Como varios idiomas-año
   quedaron por debajo de su meta, un diario puede pesar más del {args.tope:.0%} del total entregado (hasta {pct_max.max():.0f}% en algún idioma-año; ver
   `composicion_diarios.csv`). Hay que corregirlo en el análisis (por diario o por tipo de diario), no darlo por balanceado.
4. **Selección:** la mayoría de noticias se encontró porque su URL traía una palabra clave de IA; se incluyó una muestra pequeña sin ella (`control`)
   y, en diarios de tecnología, URLs sin palabra clave como relleno. Se aceptó una noticia si el texto menciona términos de IA 3 o más veces
   (o, cuando el título se guardó, si el título la mencionaba), así que puede haber algunos falsos positivos (noticias donde la IA es secundaria).
5. **Calidad del texto:** extraído automáticamente con trafilatura; puede traer restos de menús, avisos de cookies o textos truncados por muros de pago.
   Revisa `muestra_revision.csv` antes de lanzar el modelo sobre todo.
6. **Muestra, no censo:** representa lo que GDELT indexó de {d.diario.nunique()} diarios, no toda la prensa.
{nota_fecha}{nota_titulo}
## Siguientes pasos sugeridos (sentimiento y emociones)
1. **Revisar la muestra** (`muestra_revision.csv`) y decidir si hay que filtrar plantillas o textos cortos.
2. **Elegir el modelo.** Mi recuerdo, que hay que verificar en la documentación: `pysentimiento` cubre sentimiento en es/en/pt/it y emociones en
   es/en/it, es decir, **no cubre francés ni alemán**. Para esos hace falta un modelo multilingüe (por ejemplo uno tipo XLM-RoBERTa para sentimiento).
   Conviene usar **el mismo modelo en los 5 idiomas** para que los resultados sean comparables, o al menos validar con una muestra etiquetada a mano.
3. **Textos largos.** Los modelos suelen aceptar 128-512 tokens y las noticias son más largas (mediana {int(d.n_palabras.median()):,} palabras). Opciones: título + primeros
   párrafos, o dividir en fragmentos y agregar los resultados (promedio o proporción) por noticia.
4. **Ojo con el dominio:** estos modelos suelen estar entrenados con tuits; las noticias son más neutras, así que la mayoría saldrá neutral. Mejor
   analizar la distribución y los cambios relativos que la etiqueta de cada noticia.
5. **Temas:** extracción de temas por idioma (por ejemplo BERTopic con un modelo multilingüe) y luego cruzar tema × tiempo × sentimiento.
6. **Resultado esperado:** una tabla con `id` y las columnas de sentimiento/emociones/tema, para unir con este dataset por `id`.
"""
(out / "LEEME_dataset.md").write_text(leeme, encoding="utf-8")

informe = ["== VALIDACIÓN ==", *[f"  ok  {s}" for s in info], "", f"== AVISOS ({len(avisos)}) =="] + \
          [f"  !!  {s}" for s in avisos] if avisos else ["== VALIDACIÓN ==", *[f"  ok  {s}" for s in info], "", "== AVISOS == ninguno"]
(out / "validacion.txt").write_text("\n".join(informe), encoding="utf-8")
print("\n".join(informe))
print(f"\nEntrega lista en: {out.resolve()}")