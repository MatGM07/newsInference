#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preparar_datos.py  -  Preparación de las noticias para el análisis de emociones.

RESPONSABILIDAD ÚNICA: leer los 5 Parquet originales, limpiarlos, descartar
registros sin contenido útil, detectar duplicados y escribir UN archivo
consolidado:  datos/preparados/noticias_preparadas.parquet

Este script NO importa, descarga ni ejecuta ningún modelo (ni torch, ni
transformers). Los archivos originales solo se LEEN; nunca se modifican.

Uso:
    python preparar_datos.py
    python preparar_datos.py --min-palabras 40 --min-repeticiones 15
    python preparar_datos.py --limite 500      # prueba rápida (escribe en datos/preparados/prueba/)

Puede ejecutarse las veces que quieras: sobrescribe sus propias salidas.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:  # tqdm es opcional aquí: si falta, el script funciona igual, sin barra.
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:  # evita errores de codificación al imprimir tildes en consolas Windows
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------
# Rutas (siempre relativas a la ubicación de este archivo)
# --------------------------------------------------------------------------
PROYECTO = Path(__file__).resolve().parent
ORIGINALES = PROYECTO.parent                       # carpeta "entrega"
DIR_PREPARADOS = PROYECTO / "datos" / "preparados"
IDIOMAS = ("es", "en", "pt", "fr", "de")

# --------------------------------------------------------------------------
# Parámetros por defecto (todos configurables por línea de comandos)
# --------------------------------------------------------------------------
MIN_PALABRAS = 50          # tras limpiar, menos palabras => sin contenido útil
MIN_PROP_LETRAS = 0.50     # proporción mínima de letras (sobre caracteres no blancos)
MIN_REPETICIONES = 20      # una línea repetida en >= N noticias del MISMO diario es plantilla
MAX_LARGO_PLANTILLA = 300  # solo se consideran plantilla líneas de hasta N caracteres
LARGO_CLAVE_CASI_DUP = 300 # caracteres iniciales usados para marcar casi-duplicados

# --------------------------------------------------------------------------
# Expresiones regulares de limpieza
# --------------------------------------------------------------------------
# Caracteres de control, U+FFFD, marcas de ancho cero, BOM y guion blando.
RE_CONTROL = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\u200b-\u200f\u202a-\u202e"
    "\u2060\u2066-\u2069\ufeff\u00ad]"
)
RE_ESPACIOS_UNICODE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\t\f\v]")
RE_ENTIDAD = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
RE_BLOQUE_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
RE_COMENTARIO_HTML = re.compile(r"<!--.*?-->", re.S)
RE_ETIQUETA = re.compile(r"</?([A-Za-z][A-Za-z0-9:_-]*)(?:\s+[^<>]{0,300})?/?>")
ETIQUETAS_DE_BLOQUE = {"br", "p", "div", "li", "ul", "ol", "tr", "td", "th", "table",
                       "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
                       "blockquote", "hr", "figure", "figcaption", "header", "footer"}
RE_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
RE_VACIOS = re.compile(r"[\(\[\{]\s*[\)\]\}]")
RE_ESPACIOS = re.compile(r" {2,}")
RE_ALNUM = re.compile(r"[^\W_]")
RE_DIGITOS = re.compile(r"\d+")
RE_BLANCOS = re.compile(r"\s+")
RE_CODIGO = re.compile(
    r"""(?ix)
      ^\s*(?:var|let|const)\s+[\w$]+\s*=                    # var x = ...
    | \bfunction\s*[\w$]*\s*\(                              # function(...)
    | \)\s*\{\s*$                                           # ... ) {
    | ^\s*[\w.#>\s,:\[\]="'-]+\{[^{}]*:[^{}]*\}\s*$         # regla CSS
    | \{\s*"[\w@$-]+"\s*:                                   # objeto JSON
    | \b(?:window|document)\.\w+\s*(?:=|\()                 # window.x = / document.x(
    | \bdataLayer\b | \bgoogletag\b | \bgtag\s*\( | \bfbq\s*\(
    """
)
SIMBOLOS_CODIGO = "{}[]();=<>$\\|"

REGLAS_DOC = [
    ("R1 Normalización Unicode",
     "Texto en NFC; se eliminan caracteres de control, U+FFFD, marcas de ancho cero, BOM y guiones blandos; "
     "los espacios Unicode (incluido el no separable) pasan a espacio normal; los saltos de línea se unifican."),
    ("R2 Entidades y etiquetas HTML",
     "Se decodifican entidades HTML (&amp;, &quot;...). Se eliminan bloques <script>/<style>, comentarios HTML y "
     "etiquetas bien formadas (las de bloque como <br>/<p> se convierten en salto de línea para no pegar palabras). "
     "Un '<' suelto o seguido de número o espacio NO se toca."),
    ("R3 URLs",
     "Se eliminan URLs (http://, https://, www.) y los paréntesis o corchetes que queden vacíos. "
     "Una línea que solo tenía una URL desaparece."),
    ("R4 Código y datos incrustados",
     "Se descartan LÍNEAS con patrón de JavaScript/CSS/JSON (var, function(), dataLayer, objetos JSON, reglas CSS), "
     "con más de 25% de símbolos de programación (si miden >= 30 caracteres) o bloques larguísimos sin espacios "
     "(base64/minificado). Nunca se borra un párrafo de prosa."),
    ("R5 Espacios y líneas",
     "Se colapsan espacios repetidos y se quitan líneas vacías o sin ningún carácter alfanumérico. "
     "Se conservan los saltos de párrafo (una línea por párrafo) porque el modelo fragmenta por párrafos."),
    ("R6 Líneas de plantilla (basadas en frecuencia)",
     "Una línea de hasta {max_largo} caracteres que aparece idéntica (ignorando mayúsculas, espacios y cifras) en "
     ">= {min_rep} noticias DEL MISMO DIARIO se considera texto de plantilla (menús, avisos de cookies o consentimiento, "
     "muros de pago, firmas, 'Passer la publicité', etc.) y se elimina. Cada línea eliminada y su frecuencia quedan en "
     "lineas_plantilla_eliminadas.csv para que puedas auditarlas. Es una regla por diario: una frase repetida en "
     "diarios distintos no se toca."),
    ("R7 Lo que NO se hace",
     "No se eliminan títulos, subtítulos, citas ni párrafos de contenido, ni se borran palabras por mencionar "
     "'Trafilatura' (en este corpus no aparece). No se corrige ortografía, no se pasa a minúsculas, no se quitan "
     "emojis ni signos de puntuación: pueden expresar emoción. Los archivos originales no se modifican."),
    ("R8 Registros descartados",
     "Se descartan registros con texto vacío tras limpiar, con menos de {min_pal} palabras, con menos de "
     "{min_letras:.0%} de letras, con id repetido, con URL repetida o con texto idéntico (exacto, ya limpio y "
     "normalizado) a otro; de los duplicados exactos se conserva el más antiguo por fecha. Cada descarte queda en "
     "registros_descartados.csv con su motivo."),
    ("R9 Casi-duplicados (solo se marcan)",
     "Noticias del mismo diario con los mismos primeros {n} caracteres limpios se marcan con "
     "posible_duplicado=True pero NO se eliminan: pueden ser plantillas editoriales (p. ej. podcasts o boletines) "
     "o la misma nota actualizada. Decide después si quieres filtrarlas."),
]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def barra(iterable, **kw):
    return tqdm(iterable, **kw) if tqdm else iterable


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def clave_texto(s: str) -> str:
    """Clave de comparación: minúsculas, cifras->0, espacios colapsados."""
    return RE_DIGITOS.sub("0", RE_BLANCOS.sub(" ", s.casefold())).strip()


def hash_corto(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def detectar_columna_texto(df: pd.DataFrame) -> str:
    """Identifica la columna con el texto completo (sin suponer su nombre)."""
    for nombre in ("texto", "text", "contenido", "content", "cuerpo", "body"):
        if nombre in df.columns:
            return nombre
    candidatas = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not candidatas:
        raise ValueError("No encontré ninguna columna de texto en el Parquet.")
    return max(candidatas, key=lambda c: df[c].fillna("").astype(str).str.len().mean())


def _reemplazo_etiqueta(m: re.Match) -> str:
    return "\n" if m.group(1).lower() in ETIQUETAS_DE_BLOQUE else ""


def es_codigo(linea: str) -> bool:
    if RE_CODIGO.search(linea):
        return True
    n = len(linea)
    if n >= 30 and sum(linea.count(c) for c in SIMBOLOS_CODIGO) / n > 0.25:
        return True
    if n >= 200 and linea.count(" ") / n < 0.03:
        return True
    return False


def limpiar_basico(texto) -> tuple[list[str], set[str]]:
    """Reglas R1-R5. Devuelve (líneas limpias, banderas de reglas aplicadas)."""
    flags: set[str] = set()
    if not isinstance(texto, str) or not texto:
        return [], flags
    t = unicodedata.normalize("NFC", texto)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n")
    if RE_CONTROL.search(t):
        flags.add("control")
        t = RE_CONTROL.sub("", t)
    t = RE_ESPACIOS_UNICODE.sub(" ", t)
    if "&" in t and RE_ENTIDAD.search(t):
        flags.add("entidad_html")
        t = html.unescape(t)
    if "<" in t:
        t2 = RE_BLOQUE_SCRIPT.sub("\n", t)
        t2 = RE_COMENTARIO_HTML.sub("", t2)
        t2 = RE_ETIQUETA.sub(_reemplazo_etiqueta, t2)
        if t2 != t:
            flags.add("html")
            t = t2
    if "http" in t or "www." in t:
        t2 = RE_URL.sub("", t)
        if t2 != t:
            flags.add("url")
            t = RE_VACIOS.sub("", t2)
    lineas = []
    for ln in t.split("\n"):
        ln = RE_ESPACIOS.sub(" ", ln).strip()
        if not ln or not RE_ALNUM.search(ln):
            continue
        if es_codigo(ln):
            flags.add("codigo")
            continue
        lineas.append(ln)
    return lineas, flags


def normalizar_url(u) -> str:
    return u.strip() if isinstance(u, str) else ""


# --------------------------------------------------------------------------
# Procesamiento de un idioma
# --------------------------------------------------------------------------
def procesar_idioma(idioma: str, df: pd.DataFrame, columna_texto: str, args, vistos: dict, ids_muestra: set):
    st: dict = {"idioma": idioma, "filas_entrada": len(df)}
    st["palabras_originales"] = df[columna_texto].fillna("").str.split().str.len().describe(
        percentiles=[.05, .5, .95]).round(0).to_dict()

    # Normalización de campos de apoyo (sin perder los originales)
    df = df.copy()
    df["idioma"] = df["idioma"].astype(str).str.strip().str.lower() if "idioma" in df.columns else idioma
    if "diario" in df.columns:
        df["diario"] = df["diario"].astype(str).str.strip()
    if "fecha_final" in df.columns:
        df["fecha_final"] = pd.to_datetime(df["fecha_final"], errors="coerce")
        df["anio_mes"] = df["fecha_final"].dt.strftime("%Y-%m")
        df["trimestre"] = df["fecha_final"].dt.to_period("Q").astype(str)
    diarios = df["diario"].tolist() if "diario" in df.columns else ["(sin diario)"] * len(df)

    # Pasada A: limpieza básica (R1-R5)
    textos = df[columna_texto].tolist()
    lineas_por_noticia, flags_por_noticia = [], []
    for t in barra(textos, desc=f"[{idioma}] limpieza básica", unit="noticia", leave=False):
        ln, fl = limpiar_basico(t)
        lineas_por_noticia.append(ln)
        flags_por_noticia.append(fl)

    # Pasada B: plantillas por frecuencia dentro del mismo diario (R6)
    conteo: Counter = Counter()
    for d, lineas in zip(diarios, lineas_por_noticia):
        for k in {clave_texto(l) for l in lineas if len(l) <= args.max_largo_plantilla}:
            conteo[(d, k)] += 1
    plantillas = {dk for dk, n in conteo.items() if n >= args.min_repeticiones}
    ejemplos: dict = {}
    limpios, quitadas, banderas = [], [], []
    for d, lineas, fl in zip(diarios, lineas_por_noticia, flags_por_noticia):
        nuevas, q = [], 0
        for l in lineas:
            if len(l) <= args.max_largo_plantilla:
                k = (d, clave_texto(l))
                if k in plantillas:
                    q += 1
                    ejemplos.setdefault(k, l)
                    continue
            nuevas.append(l)
        if q:
            fl = fl | {"plantilla"}
        limpios.append("\n".join(nuevas))
        quitadas.append(q)
        banderas.append(",".join(sorted(fl)))
    st["lineas_plantilla_tipos"] = len(plantillas)
    st["lineas_plantilla_quitadas"] = int(sum(quitadas))
    filas_plantilla = [
        {"idioma": idioma, "diario": d, "linea_ejemplo": ejemplos[(d, k)],
         "n_noticias": conteo[(d, k)], "n_caracteres": len(ejemplos[(d, k)])}
        for (d, k) in plantillas if (d, k) in ejemplos
    ]

    # Métricas y columnas nuevas
    df["texto_original"] = df[columna_texto]
    if columna_texto != "texto_original":
        df = df.drop(columns=[columna_texto])
    df["texto_limpio"] = limpios
    df["n_palabras_limpio"] = df["texto_limpio"].str.split().str.len().astype("int64")
    df["n_caracteres_limpio"] = df["texto_limpio"].str.len().astype("int64")
    df["n_lineas_plantilla"] = quitadas
    df["limpieza_flags"] = banderas

    def prop_letras(t: str) -> float:
        base = len(t) - t.count(" ") - t.count("\n")
        return (sum(map(str.isalpha, t)) / base) if base > 0 else 0.0

    # Cambios de limpieza vistos en la muestra de revisión
    revision = []
    if ids_muestra:
        for i in df.index[df["id"].isin(ids_muestra)]:
            r = df.loc[i]
            revision.append({
                "id": r["id"], "idioma": idioma, "diario": r.get("diario"),
                "n_palabras_original": int(len(str(r["texto_original"]).split())),
                "n_palabras_limpio": int(r["n_palabras_limpio"]),
                "n_lineas_plantilla": int(r["n_lineas_plantilla"]),
                "limpieza_flags": r["limpieza_flags"],
                "inicio_original": str(r["texto_original"])[:300].replace("\n", " ⏎ "),
                "inicio_limpio": str(r["texto_limpio"])[:300].replace("\n", " ⏎ "),
            })

    # Descartes (R8)
    motivos = pd.Series("", index=df.index, dtype="object")
    motivos[df["texto_limpio"].str.strip() == ""] = "texto_vacio_tras_limpieza"
    motivos[(motivos == "") & (df["n_palabras_limpio"] < args.min_palabras)] = f"menos_de_{args.min_palabras}_palabras"
    mask_letras = (motivos == "")
    props = df.loc[mask_letras, "texto_limpio"].map(prop_letras)
    motivos[props[props < args.min_prop_letras].index] = "baja_proporcion_de_letras"

    # Orden determinista: se conserva la noticia más antigua entre duplicados exactos
    orden = df.sort_values(["fecha_final", "id"] if "fecha_final" in df.columns else ["id"]).index
    for i in orden:
        if motivos[i]:
            continue
        r_id = df.at[i, "id"]
        r_url = normalizar_url(df.at[i, "url"]) if "url" in df.columns else ""
        h = hash_corto(clave_texto(df.at[i, "texto_limpio"]))
        if r_id in vistos["id"]:
            motivos[i] = "id_duplicado"
        elif r_url and r_url in vistos["url"]:
            motivos[i] = "url_duplicada"
        elif h in vistos["texto"]:
            motivos[i] = "duplicado_exacto"
        else:
            vistos["id"].add(r_id)
            if r_url:
                vistos["url"].add(r_url)
            vistos["texto"].add(h)

    descartados = df.loc[motivos != "", [c for c in ("id", "idioma", "diario", "url") if c in df.columns]].copy()
    descartados["motivo"] = motivos[motivos != ""]
    st["descartes"] = descartados["motivo"].value_counts().to_dict()
    df = df[motivos == ""].copy()

    # Casi-duplicados (R9): solo marcar
    claves = df["texto_limpio"].map(lambda t: clave_texto(t)[:args.largo_casi_dup] if len(t) >= args.largo_casi_dup else "")
    grupo = (df["diario"].astype(str) + "|" + claves) if "diario" in df.columns else claves
    tam = grupo.map(grupo.value_counts())
    df["posible_duplicado"] = ((claves != "") & (tam >= 2)).values
    st["posibles_duplicados"] = int(df["posible_duplicado"].sum())
    st["filas_salida"] = len(df)
    st["palabras_limpias"] = df["n_palabras_limpio"].describe(percentiles=[.05, .5, .95]).round(0).to_dict()
    st["banderas"] = Counter(f for fl in df["limpieza_flags"] for f in fl.split(",") if f)
    st["banderas_antes_descarte"] = Counter(f for fl in banderas for f in fl.split(",") if f)
    top = df.groupby("diario")["n_lineas_plantilla"].sum().sort_values(ascending=False).head(8) if "diario" in df.columns else pd.Series(dtype=int)
    st["top_diarios_plantilla"] = {k: int(v) for k, v in top.items() if v > 0}
    st["lineas_chicas_ejemplo"] = []
    return df, st, filas_plantilla, descartados, revision


# --------------------------------------------------------------------------
# Validación contra los CSV complementarios
# --------------------------------------------------------------------------
def validar_auxiliares(meta: pd.DataFrame, originales: Path) -> list[str]:
    msgs = []

    def comparar(nombre, esperado: pd.DataFrame, real: pd.DataFrame, claves, col_esp):
        m = real.merge(esperado, on=claves, how="outer")
        ok = bool((m["n_real"].fillna(-1) == m[col_esp].fillna(-1)).all())
        msgs.append(f"- `{nombre}`: {'coincide exactamente' if ok else '**DIFIERE**'} con el conteo de los Parquet "
                    f"({len(m)} combinaciones).")

    try:
        f = originales / "cobertura_idioma_anio.csv"
        if f.exists():
            e = pd.read_csv(f).rename(columns={"anio": "anio_pub"})
            r = meta.groupby(["idioma", "anio_pub"]).size().reset_index(name="n_real")
            comparar(f.name, e[["idioma", "anio_pub", "final"]], r, ["idioma", "anio_pub"], "final")
        f = originales / "composicion_diarios.csv"
        if f.exists():
            e = pd.read_csv(f)
            r = meta.groupby(["idioma", "anio_gdelt", "diario"]).size().reset_index(name="n_real")
            comparar(f.name, e[["idioma", "anio_gdelt", "diario", "n"]], r, ["idioma", "anio_gdelt", "diario"], "n")
        f = originales / "serie_mensual.csv"
        if f.exists():
            e = pd.read_csv(f).rename(columns={"ym": "anio_mes"})
            r = meta.groupby(["idioma", "anio_mes"]).size().reset_index(name="n_real")
            comparar(f.name, e[["idioma", "anio_mes", "n"]], r, ["idioma", "anio_mes"], "n")
    except Exception as exc:  # la validación nunca debe tumbar la preparación
        msgs.append(f"- No pude completar la validación con los CSV complementarios: {exc}")
    return msgs


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Prepara las noticias para el análisis de emociones (NO ejecuta el modelo).")
    ap.add_argument("--originales", type=Path, default=ORIGINALES, help="Carpeta con los Parquet originales.")
    ap.add_argument("--min-palabras", type=int, default=MIN_PALABRAS)
    ap.add_argument("--min-prop-letras", type=float, default=MIN_PROP_LETRAS)
    ap.add_argument("--min-repeticiones", type=int, default=MIN_REPETICIONES)
    ap.add_argument("--max-largo-plantilla", type=int, default=MAX_LARGO_PLANTILLA)
    ap.add_argument("--largo-casi-dup", type=int, default=LARGO_CLAVE_CASI_DUP)
    ap.add_argument("--limite", type=int, default=None,
                    help="Procesa solo N filas por idioma y escribe en datos/preparados/prueba/ (no toca la salida real).")
    args = ap.parse_args()

    originales = args.originales.resolve()
    destino = DIR_PREPARADOS / "prueba" if args.limite else DIR_PREPARADOS
    for d in (destino, PROYECTO / "resultados", PROYECTO / "modelos"):
        d.mkdir(parents=True, exist_ok=True)
    salida = destino / "noticias_preparadas.parquet"
    tmp = destino / "noticias_preparadas.parquet.tmp"

    archivos = {i: originales / f"noticias_{i}.parquet" for i in IDIOMAS}
    faltan = [str(p) for p in archivos.values() if not p.exists()]
    if faltan:
        print("ERROR: no encuentro estos archivos originales:\n  " + "\n  ".join(faltan))
        print(f"Carpeta buscada: {originales}")
        return 1

    t0 = time.time()
    print(f"Carpeta de originales : {originales}")
    print(f"Salida                : {salida}")
    if args.limite:
        print(f"*** MODO PRUEBA: solo {args.limite} filas por idioma ***")

    entradas = []
    for i, p in archivos.items():
        print(f"Calculando huella de {p.name} ...", end=" ", flush=True)
        entradas.append({"archivo": p.name, "bytes": p.stat().st_size, "sha256": sha256_archivo(p)})
        print("ok")

    # Muestra de revisión (solo para auditar la limpieza)
    ids_muestra: set = set()
    f_mu = originales / "muestra_revision.csv"
    if f_mu.exists() and not args.limite:
        ids_muestra = set(pd.read_csv(f_mu, usecols=["id"])["id"].astype(str))

    vistos = {"id": set(), "url": set(), "texto": set()}
    writer = None
    schema_ref = None
    stats, plantillas_all, descartes_all, revision_all, metas = [], [], [], [], []
    ids_todos: set = set()
    columnas_originales = None
    columna_texto = None

    try:
        for idioma in IDIOMAS:
            print(f"\n=== Idioma: {idioma} ===")
            df = pd.read_parquet(archivos[idioma])
            if args.limite:
                df = df.head(args.limite)
            if columnas_originales is None:
                columnas_originales = list(df.columns)
                columna_texto = detectar_columna_texto(df)
                print(f"Columnas ({len(columnas_originales)}): {columnas_originales}")
                print(f"Columna de texto detectada: '{columna_texto}'")
            elif list(df.columns) != columnas_originales:
                print(f"ADVERTENCIA: columnas distintas en {idioma}; se usan las comunes.")
            if "idioma" in df.columns and (df["idioma"].astype(str).str.lower() != idioma).any():
                print(f"ADVERTENCIA: {archivos[idioma].name} contiene filas con otro valor en 'idioma'; se conserva el valor de la fila.")
            ids_todos.update(df["id"].astype(str))
            cols_meta = [c for c in ("idioma", "anio_pub", "anio_gdelt", "diario", "fecha_final") if c in df.columns]
            mm = df[cols_meta].copy()
            mm["anio_mes"] = pd.to_datetime(mm["fecha_final"]).dt.strftime("%Y-%m")
            metas.append(mm)

            out, st, plant, desc, rev = procesar_idioma(idioma, df, columna_texto, args, vistos, ids_muestra)
            stats.append(st)
            plantillas_all += plant
            descartes_all.append(desc)
            revision_all += rev
            print(f"  entrada {st['filas_entrada']:,} -> salida {st['filas_salida']:,} | "
                  f"descartes {st['descartes']} | líneas de plantilla quitadas {st['lineas_plantilla_quitadas']:,}")

            # Orden de columnas: primero las habituales, luego el resto de metadatos originales
            front = [c for c in ("id", "idioma", "diario", "tipo", "rol", "fecha_final", "anio_pub", "mes_pub",
                                 "anio_mes", "trimestre", "titulo", "texto_original", "texto_limpio") if c in out.columns]
            resto = [c for c in out.columns if c not in front]
            out = out[front + resto].sort_values(["fecha_final", "id"] if "fecha_final" in out.columns else ["id"])
            tabla = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                parametros = {"min_palabras": args.min_palabras, "min_prop_letras": args.min_prop_letras,
                              "min_repeticiones": args.min_repeticiones,
                              "max_largo_plantilla": args.max_largo_plantilla,
                              "columna_texto_para_el_modelo": "texto_limpio", "limite": args.limite}
                schema_ref = tabla.schema.with_metadata({
                    b"preparar_datos": json.dumps(parametros, ensure_ascii=False).encode("utf-8")})
                writer = pq.ParquetWriter(tmp, schema_ref, compression="zstd")
            writer.write_table(tabla.cast(schema_ref))
            del df, out, tabla
    except Exception:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise
    writer.close()
    os.replace(tmp, salida)

    # --- Reportes -----------------------------------------------------------
    meta_df = pd.concat(metas, ignore_index=True)
    mensajes_aux = validar_auxiliares(meta_df, originales) if not args.limite else ["- (omitida en modo prueba)"]

    pd.DataFrame(plantillas_all).sort_values("n_noticias", ascending=False).to_csv(
        destino / "lineas_plantilla_eliminadas.csv", index=False, encoding="utf-8-sig") if plantillas_all else None
    pd.concat(descartes_all, ignore_index=True).to_csv(
        destino / "registros_descartados.csv", index=False, encoding="utf-8-sig")
    if revision_all:
        pd.DataFrame(revision_all).to_csv(destino / "revision_limpieza_muestra.csv", index=False, encoding="utf-8-sig")

    final = pq.ParquetFile(salida)
    total_in = sum(s["filas_entrada"] for s in stats)
    total_out = sum(s["filas_salida"] for s in stats)
    resumen = pd.read_parquet(salida, columns=[c for c in ("idioma", "anio_pub", "tipo") if c in final.schema_arrow.names])
    if {"idioma", "anio_pub"} <= set(resumen.columns):
        resumen.groupby(["idioma", "anio_pub"]).size().reset_index(name="n_noticias_preparadas").to_csv(
            destino / "resumen_idioma_anio.csv", index=False, encoding="utf-8-sig")

    reglas_txt = "\n".join(
        f"- **{t}.** " + d.format(max_largo=args.max_largo_plantilla, min_rep=args.min_repeticiones,
                                 min_pal=args.min_palabras, min_letras=args.min_prop_letras, n=args.largo_casi_dup)
        for t, d in REGLAS_DOC)

    L = ["# Informe de calidad de la preparación de datos", "",
         f"Generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
         f"Duración: {time.time() - t0:.0f} s  ", f"Salida: `{salida.name}` ({salida.stat().st_size/1e6:.1f} MB)", "",
         f"**Noticias de entrada: {total_in:,} → preparadas: {total_out:,} ({total_in - total_out:,} descartadas, "
         f"{(total_in - total_out) / max(total_in,1):.1%}).**", "",
         "## Resumen por idioma", "",
         "| idioma | entrada | salida | descartes (motivo: n) | líneas de plantilla quitadas | posibles duplicados (marcados) |",
         "|---|---|---|---|---|---|"]
    for s in stats:
        L.append(f"| {s['idioma']} | {s['filas_entrada']:,} | {s['filas_salida']:,} | "
                 f"{', '.join(f'{k}: {v}' for k, v in s['descartes'].items()) or '-'} | "
                 f"{s['lineas_plantilla_quitadas']:,} | {s['posibles_duplicados']:,} |")
    L += ["", "## Palabras por noticia (mediana / p5 / p95)", "", "| idioma | original | limpio |", "|---|---|---|"]
    for s in stats:
        o, c = s["palabras_originales"], s["palabras_limpias"]
        L.append(f"| {s['idioma']} | {o['50%']:.0f} / {o['5%']:.0f} / {o['95%']:.0f} | {c['50%']:.0f} / {c['5%']:.0f} / {c['95%']:.0f} |")
    L += ["", "## Noticias afectadas por cada regla (antes de descartar)", "",
          "| idioma | " + " | ".join(["control", "entidad_html", "html", "url", "codigo", "plantilla"]) + " |",
          "|---|" + "---|" * 6]
    for s in stats:
        b = s["banderas_antes_descarte"]
        L.append(f"| {s['idioma']} | " + " | ".join(f"{b.get(k, 0):,}" for k in ["control", "entidad_html", "html", "url", "codigo", "plantilla"]) + " |")
    L += ["", "## Diarios con más líneas de plantilla eliminadas", ""]
    for s in stats:
        if s["top_diarios_plantilla"]:
            L.append(f"- **{s['idioma']}**: " + ", ".join(f"{k} ({v:,})" for k, v in s["top_diarios_plantilla"].items()))
    L += ["", "## Verificación contra los archivos complementarios", ""] + mensajes_aux
    L += ["", "## Reglas de limpieza aplicadas", "", reglas_txt, "",
          "## Parámetros", "",
          f"`min_palabras={args.min_palabras}`, `min_prop_letras={args.min_prop_letras}`, "
          f"`min_repeticiones={args.min_repeticiones}`, `max_largo_plantilla={args.max_largo_plantilla}`", "",
          "## Columnas del archivo preparado", "", ", ".join(f"`{n}`" for n in final.schema_arrow.names), "",
          "El modelo debe leer la columna **`texto_limpio`**; `texto_original` se conserva intacta para auditoría."]
    (destino / "informe_calidad.md").write_text("\n".join(L), encoding="utf-8")

    meta_json = {
        "generado_utc": datetime.now(timezone.utc).isoformat(), "duracion_s": round(time.time() - t0, 1),
        "python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pa.__version__,
        "parametros": {"min_palabras": args.min_palabras, "min_prop_letras": args.min_prop_letras,
                       "min_repeticiones": args.min_repeticiones, "max_largo_plantilla": args.max_largo_plantilla,
                       "largo_casi_dup": args.largo_casi_dup, "limite": args.limite},
        "columna_texto_original_detectada": columna_texto, "columna_texto_para_el_modelo": "texto_limpio",
        "entradas": entradas, "filas_entrada": total_in, "filas_salida": total_out,
        "sha256_salida": sha256_archivo(salida),
    }
    (destino / "preparacion_meta.json").write_text(json.dumps(meta_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nListo en {time.time() - t0:.0f} s: {total_in:,} -> {total_out:,} noticias.")
    print(f"Archivo preparado : {salida}")
    print(f"Informe de calidad: {destino / 'informe_calidad.md'}")
    print("Nota: este script NO ejecutó el modelo. Para eso: python ejecutar_modelo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
