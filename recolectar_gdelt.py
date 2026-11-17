#!/usr/bin/env python3
"""
Recolección completa de noticias de IA (GDELT + trafilatura) en tres pasos reanudables.

  python recolectar_gdelt.py urls --proyecto TU_PROYECTO_GCP   # 1) URLs mensuales desde GDELT
  python recolectar_gdelt.py descargar                         # 2) descarga hasta cumplir metas.csv
  python recolectar_gdelt.py consolidar                        # 3) dataset final + informe

Cada paso se puede interrumpir (Ctrl+C) y volver a lanzar: retoma donde quedó.

Cómo se llenan las metas (metas.csv: una meta por idioma y año):
  - Para cada idioma-año se arma una cola aleatoria con las URLs de GDELT de esos meses.
    Las URLs con palabra clave de IA entran con peso 1; las URLs sin ella de los diarios de
    tecnología, con --peso-tech (0.3); y hay una pequeña muestra de control de los
    generalistas (URLs sin palabra clave) para medir qué se pierde por filtrar por URL.
  - Se descarga por lotes y solo se acepta una noticia si: el texto se extrae, trata de IA
    (título con término de IA o >= --min-menciones en el texto), tiene >= --min-palabras
    y el idioma detectado coincide con el del diario.
  - Pasada 1: ningún diario pasa de --tope (30%) de la meta. Pasada 2: sin tope, solo si la
    meta no se cumplió. Pasada 3: diarios de reserva, solo si aún falta.
Salidas en salida/: urls/, articulos/<idioma>_<año>/parte_*.parquet, dataset_noticias.parquet
"""
import argparse
import hashlib
import json
import math
import re
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import trafilatura
from google.cloud import bigquery
from langdetect import DetectorFactory, detect
from tqdm import tqdm

from piloto_gdelt import CONFIG, URL_PATRON, desde_wayback, esperar, menciones_ia

DetectorFactory.seed = 0
SALIDA = Path("salida")
DIR_URLS = SALIDA / "urls"
DIR_ARTS = SALIDA / "articulos"


# ============================================================ configuración
def validar(s: str) -> str:
    """Los valores van dentro del SQL: solo se permiten caracteres de dominio/ruta."""
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", s):
        raise ValueError(f"Valor no permitido en diarios.csv: {s!r}")
    return s


def cargar_diarios(ruta) -> pd.DataFrame:
    d = pd.read_csv(ruta)
    d["requiere_url"] = d["requiere_url"].fillna("") if "requiere_url" in d else ""
    d["rol"] = d["rol"].fillna("principal") if "rol" in d else "principal"
    return d


def mapa_alias(d: pd.DataFrame) -> dict:
    return {validar(x.strip()): r.diario for r in d.itertuples()
            for x in r.dominios_gdelt.split(";")}


# ============================================================ paso 1: URLs
def sql_mes(d: pd.DataFrame) -> str:
    req = []
    for r in d.itertuples():
        if r.requiere_url:
            for dom in r.dominios_gdelt.split(";"):
                req.append(f"(SourceCommonName != '{validar(dom.strip())}' "
                           f"OR STRPOS(DocumentIdentifier, '{validar(r.requiere_url)}') > 0)")
    extra = " AND ".join(req) if req else "TRUE"
    return f"""
WITH base AS (
  SELECT DISTINCT
    SourceCommonName AS dominio,
    DocumentIdentifier AS url,
    SUBSTR(CAST(DATE AS STRING), 1, 8) AS fecha
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP(@ini) AND _PARTITIONTIME < TIMESTAMP(@fin)
    AND SourceCommonName IN UNNEST(@dominios)
    AND {extra}
),
marc AS (
  SELECT *, REGEXP_CONTAINS(LOWER(url), @patron) AS match_url FROM base
),
ctrl AS (
  SELECT * EXCEPT(rn) FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY dominio ORDER BY FARM_FINGERPRINT(url)) AS rn
    FROM marc
    WHERE NOT match_url AND dominio NOT IN UNNEST(@tech)
  ) WHERE rn <= @control
)
SELECT * FROM marc WHERE match_url OR dominio IN UNNEST(@tech)
UNION ALL
SELECT * FROM ctrl
"""


def cmd_urls(args):
    d = cargar_diarios(args.diarios)
    alias = mapa_alias(d)
    tipo = d.set_index("diario")["tipo"]
    tech = sorted(dom for dom, k in alias.items() if tipo[k] == "tecnologia")
    sql = sql_mes(d)
    client = bigquery.Client(project=args.proyecto)

    def params(m):
        ini = pd.Timestamp(f"{m}-01")
        fin = ini + pd.offsets.MonthBegin(1)
        return [
            bigquery.ScalarQueryParameter("patron", "STRING", URL_PATRON),
            bigquery.ScalarQueryParameter("ini", "STRING", ini.strftime("%Y-%m-%d")),
            bigquery.ScalarQueryParameter("fin", "STRING", fin.strftime("%Y-%m-%d")),
            bigquery.ScalarQueryParameter("control", "INT64", args.control),
            bigquery.ArrayQueryParameter("dominios", "STRING", sorted(alias)),
            bigquery.ArrayQueryParameter("tech", "STRING", tech),
        ]

    meses = [str(p) for p in pd.period_range(args.desde, args.hasta, freq="M")]
    pendientes = [m for m in meses if not (DIR_URLS / f"{m}.parquet").exists()]
    print(f"{len(meses) - len(pendientes)} meses ya descargados, {len(pendientes)} pendientes.")
    if not pendientes:
        return

    # La simulación valida el SQL y estima el costo antes de gastar cuota.
    seco = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params(pendientes[0]), dry_run=True, use_query_cache=False))
    gb = seco.total_bytes_processed / 1e9
    print(f"~{gb:.1f} GB por mes -> ~{gb * len(pendientes):.0f} GB en total "
          f"(capa gratuita: 1 TB al mes).")
    if gb * len(pendientes) > args.max_gb and not args.si:
        if input("Supera el límite configurado. ¿Continuar? [s/N] ").strip().lower() != "s":
            sys.exit("Cancelado.")

    DIR_URLS.mkdir(parents=True, exist_ok=True)
    for m in tqdm(pendientes, desc="Meses"):
        df = client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=params(m))).to_dataframe()
        df["mes_gdelt"] = m
        df["diario"] = df["dominio"].map(alias)
        df.to_parquet(DIR_URLS / f"{m}.parquet", index=False)
    print(f"Listo. URLs en {DIR_URLS.resolve()}")


# ============================================================ paso 2: descarga
def idioma_texto(texto: str):
    try:
        return detect(texto[:1000])
    except Exception:
        return None


def bajar(f: dict, args) -> dict:
    res = {**f, "via": None, "titulo": None, "fecha_pub": None, "texto": None,
           "n_palabras": 0, "n_menciones_ia": 0, "es_ia": False,
           "idioma_detectado": None, "aceptada": False, "error": None}
    try:
        esperar(urlparse(f["url"]).netloc, args.delay)
        html, via = trafilatura.fetch_url(f["url"], config=CONFIG), "directo"
        if not html and not args.sin_wayback:
            html, via = desde_wayback(f["url"], f["fecha"]), "wayback"
        if not html:
            res["error"] = "sin_descarga"
            return res
        salida = trafilatura.extract(html, output_format="json", include_comments=False,
                                     include_tables=False, favor_precision=True, config=CONFIG)
        if not salida:
            res["error"] = "sin_texto"
            return res
        d = json.loads(salida)
        texto, titulo = d.get("text") or "", d.get("title") or ""
        n = menciones_ia(texto)
        es_ia = menciones_ia(titulo) > 0 or n >= args.min_menciones
        n_pal = len(texto.split())
        res.update(via=via, titulo=titulo, fecha_pub=d.get("date"), n_palabras=n_pal,
                   n_menciones_ia=n, es_ia=bool(es_ia))
        if not es_ia:
            res["error"] = "no_ia"
        elif n_pal < args.min_palabras:
            res["error"] = "corta"
        else:
            lang = idioma_texto(texto)
            res["idioma_detectado"] = lang
            if lang != f["idioma"]:
                res["error"] = "otro_idioma"
            else:
                res.update(aceptada=True, texto=texto)
    except Exception as e:  # un fallo no debe tumbar la corrida
        res["error"] = f"excepcion:{type(e).__name__}"
    return res


class Estado:
    """Lo ya intentado y aceptado de un idioma-año, reconstruido desde disco."""

    def __init__(self, carpeta: Path):
        self.carpeta = carpeta
        carpeta.mkdir(parents=True, exist_ok=True)
        self.vistas, self.por_diario, self.total, self.partes = set(), Counter(), 0, 0
        for p in sorted(carpeta.glob("parte_*.parquet")):
            x = pd.read_parquet(p, columns=["url", "diario", "aceptada"])
            self.vistas.update(x.url)
            self.por_diario.update(x[x.aceptada].diario.tolist())
            self.total += int(x.aceptada.sum())
            self.partes += 1

    def registrar(self, resultados: list):
        pd.DataFrame(resultados).to_parquet(
            self.carpeta / f"parte_{self.partes:05d}.parquet", index=False)
        self.partes += 1
        for r in resultados:
            self.vistas.add(r["url"])
            if r["aceptada"]:
                self.por_diario[r["diario"]] += 1
                self.total += 1


def cargar_pool(diarios: pd.DataFrame, anios=None, idiomas=None) -> pd.DataFrame:
    archivos = sorted(DIR_URLS.glob("*.parquet"))
    if anios is not None:  # los archivos se llaman AAAA-MM.parquet: solo se leen los años pedidos
        archivos = [a for a in archivos if int(a.stem[:4]) in anios]
    if not archivos:
        sys.exit("No hay URLs. Corre primero: python recolectar_gdelt.py urls --proyecto ...")
    print(f"Cargando {len(archivos)} archivos de URLs (puede tardar un minuto, no lo interrumpas)...")
    pool = pd.concat((pd.read_parquet(a) for a in archivos), ignore_index=True)
    pool = pool.join(diarios.set_index("diario")[["idioma", "tipo", "rol"]], on="diario")
    pool = pool.dropna(subset=["idioma"])
    if idiomas is not None:
        pool = pool[pool.idioma.isin(idiomas)]
    pool = pool.drop_duplicates("url")
    pool["anio"] = pool.mes_gdelt.str[:4].astype(int)
    pool["origen_url"] = np.where(pool.match_url, "match",
                                  np.where(pool.tipo == "tecnologia", "tech_sin_match", "control"))
    return pool.sort_values("url").reset_index(drop=True)


def ordenar(df: pd.DataFrame, args, semilla: int) -> list:
    """Orden aleatorio ponderado reproducible (Efraimidis-Spirakis)."""
    if df.empty:
        return []
    # Peso por URL con match = 1. Las URLs sin match (tecnología o control) se ponderan
    # POR DIARIO para que, en conjunto, pesen `peso_*` veces las URLs con match de ese diario.
    # (Antes pesaban 0.3 cada una y, al haber ~50 veces más URLs sin match que con match,
    #  se comían casi todas las descargas con una tasa de acierto de ~4%.)
    n_match = df[df.origen_url == "match"].groupby("diario").size()
    n_otro = df[df.origen_url != "match"].groupby("diario").size()
    base = df.diario.map(n_match).fillna(0).clip(lower=10)
    otro = df.diario.map(n_otro).fillna(1)
    cuota = df.origen_url.map({"tech_sin_match": args.peso_tech,
                               "control": args.peso_control}).fillna(1.0)
    w = np.where(df.origen_url == "match", 1.0, cuota * base / otro)
    u = np.random.default_rng(semilla).random(len(df))
    u = np.maximum(u, 1e-15)  # Evita el log(0)
    clave = np.log(u) / w
    return df.iloc[np.argsort(-clave)].to_dict("records")


def llenar(cola: list, meta: int, cap, estado: Estado, args, barra, limite: int):
    i = 0
    with ThreadPoolExecutor(args.workers) as ex:
        while estado.total < meta and i < len(cola) and len(estado.vistas) < limite:
            lote = []
            while i < len(cola) and len(lote) < args.lote:
                f = cola[i]
                i += 1
                if f["url"] in estado.vistas:
                    continue
                if cap is not None and estado.por_diario[f["diario"]] >= cap:
                    continue
                lote.append(f)
            if not lote:
                break
            antes = estado.total
            estado.registrar(list(ex.map(lambda f: bajar(f, args), lote)))
            barra.update(estado.total - antes)


def cmd_descargar(args):
    diarios = cargar_diarios(args.diarios)
    metas = pd.read_csv(args.metas)
    if args.idiomas:
        metas = metas[metas.idioma.isin(args.idiomas.split(","))]
    if args.anios:
        metas = metas[metas.anio.isin([int(a) for a in args.anios.split(",")])]
    # Solo se carga lo que este proceso va a usar (menos RAM y arranque más rápido).
    pool = cargar_pool(diarios, set(metas.anio), set(metas.idioma))
    print(f"URLs en el pool: {len(pool):,}")

    for m in metas.sort_values(["anio", "idioma"]).itertuples():
        b = pool[(pool.idioma == m.idioma) & (pool.anio == m.anio)]
        meta = int(m.meta)
        estado = Estado(DIR_ARTS / f"{m.idioma}_{m.anio}")
        if estado.total >= meta:
            print(f"{m.idioma} {m.anio}: completo ({estado.total}/{meta})")
            continue
        if b.empty:
            print(f"{m.idioma} {m.anio}: sin URLs en el pool")
            continue
        semilla = args.semilla + zlib.crc32(f"{m.idioma}{m.anio}".encode()) % 100000
        cap = math.ceil(args.tope * meta)
        princ = ordenar(b[b.rol == "principal"], args, semilla)
        reserva = ordenar(b[b.rol == "reserva"], args, semilla + 1)
        limite = int(args.max_intentos * meta)  # evita masticar toda la cola con tasas bajas
        with tqdm(total=meta, initial=estado.total, desc=f"{m.idioma} {m.anio}") as barra:
            llenar(princ, meta, cap, estado, args, barra, limite)
            llenar(princ, meta, None, estado, args, barra, limite)
            if estado.total < meta:
                llenar(reserva, meta, cap, estado, args, barra, limite)
                llenar(reserva, meta, None, estado, args, barra, limite)
        if estado.total >= meta:
            marca = "OK"
        elif len(estado.vistas) >= limite:
            marca = "LIMITE_INTENTOS"
        else:
            marca = "AGOTADA"
        print(f"{m.idioma} {m.anio}: {estado.total}/{meta} [{marca}] "
              f"intentos={len(estado.vistas):,} por_diario={dict(estado.por_diario.most_common(3))}")


# ============================================================ paso 3: consolidar
def recortar(g: pd.DataFrame, meta: int, tope: float, semilla: int) -> pd.DataFrame:
    g = g.sample(frac=1, random_state=semilla)
    cap = math.ceil(tope * meta)
    rango = g.groupby("diario").cumcount()
    dentro = g[rango < cap]
    if len(dentro) >= meta:
        return dentro.head(meta)
    return pd.concat([dentro, g[rango >= cap].head(meta - len(dentro))])


def cmd_consolidar(args):
    metas = pd.read_csv(args.metas)
    partes = sorted(DIR_ARTS.glob("*/parte_*.parquet"))
    if not partes:
        sys.exit("No hay descargas. Corre primero: python recolectar_gdelt.py descargar")
    todo = pd.concat((pd.read_parquet(p) for p in partes), ignore_index=True)

    print("\n== Resultado de los intentos de descarga ==")
    print(todo.error.fillna("aceptada").value_counts().to_string())
    tasa = (todo.groupby(["idioma", "origen_url"]).aceptada.agg(["size", "mean"]).round(3)
            .rename(columns={"size": "intentos", "mean": "tasa_aceptacion"}))
    print("\n== Tasa de aceptación por idioma y origen de la URL ==")
    print("(control = URLs sin palabra clave de IA: indica qué tan ciego es el filtro por URL)")
    print(tasa.to_string())

    ok = todo[todo.aceptada].copy()
    d0 = args.huella_desde  # 0 = inicio del texto; 300 salta el arranque (avisos, plantillas)
    ok["huella"] = (ok.texto.str[d0:d0 + 600].str.lower().str.replace(r"\W+", "", regex=True)
                    .map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()))
    print(f"\nDeduplicación (aceptadas: {len(ok):,}):")
    n0 = len(ok)
    ok = ok.drop_duplicates("url")
    print(f"  misma URL:                      -{n0 - len(ok):,}")
    # Solo se compara el título cuando existe: si viene vacío, todas las noticias de un diario
    # "se repetirían" entre sí y se borraría casi todo.
    n0 = len(ok)
    con_titulo = ok.titulo.fillna("").str.strip() != ""
    ok = ok[~(ok.duplicated(["diario", "titulo"]) & con_titulo)]
    print(f"  mismo diario y título (no vacío): -{n0 - len(ok):,}   (títulos disponibles: {con_titulo.mean():.0%})")
    n0 = len(ok)
    if not args.sin_huella:
        ok = ok.drop_duplicates("huella")
    quitadas = n0 - len(ok)
    print(f"  mismo inicio de texto:          -{quitadas:,}")
    if quitadas > 0.03 * n0:
        print("  AVISO: más del 3% por 'mismo inicio de texto'. Puede ser texto de plantilla común a un diario\n"
              "  (cookies, avisos) y no duplicados reales. Corre revisar_duplicados.py para saberlo.")
    print(f"  quedan: {len(ok):,}")

    trozos = []
    for (idioma, anio), g in ok.groupby(["idioma", "anio"]):
        fila = metas[(metas.idioma == idioma) & (metas.anio == anio)]
        meta = int(fila.meta.iloc[0]) if len(fila) else len(g)
        if args.cap_estricto:
            # Aplica el tope por diario aunque el resultado quede por debajo de la meta.
            cap = math.ceil(args.tope * meta)
            g = g.sample(frac=1, random_state=args.semilla)
            g = g[g.groupby("diario").cumcount() < cap]
        trozos.append(g if len(g) <= meta else recortar(g, meta, args.tope, args.semilla))
    final = pd.concat(trozos, ignore_index=True)

    ini, fin = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-10-31")
    fp = pd.to_datetime(final.fecha_pub, errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    fg = pd.to_datetime(final.fecha, format="%Y%m%d", errors="coerce")
    valida = fp.between(ini, fin)
    final["fecha_final"] = fp.where(valida, fg)
    final["fecha_origen"] = np.where(valida, "pagina", "gdelt")
    final["anio_pub"] = final.fecha_final.dt.year
    final["mes_pub"] = final.fecha_final.dt.month
    final["id"] = final.url.map(lambda u: hashlib.md5(u.encode()).hexdigest()[:12])
    final = final.rename(columns={"anio": "anio_gdelt"})
    cols = ["id", "idioma", "diario", "tipo", "rol", "anio_gdelt", "fecha_final", "anio_pub",
            "mes_pub", "fecha_origen", "titulo", "texto", "n_palabras", "n_menciones_ia",
            "origen_url", "via", "url"]
    final = final[cols].sort_values(["fecha_final", "diario"]).reset_index(drop=True)

    SALIDA.mkdir(exist_ok=True)
    final.to_parquet(SALIDA / "dataset_noticias.parquet", index=False)
    final.drop(columns="texto").sample(min(1000, len(final)), random_state=1) \
        .to_csv(SALIDA / "muestra_1000.csv", index=False)

    tabla = final.pivot_table(index="idioma", columns="anio_gdelt", values="id",
                              aggfunc="count", fill_value=0)
    objetivo = metas.pivot_table(index="idioma", columns="anio", values="meta", aggfunc="sum")
    cmp = (tabla.reindex_like(objetivo).fillna(0).astype(int).astype(str) + "/"
           + objetivo.fillna(0).astype(int).astype(str))
    print("\n== Final / meta por idioma y año ==")
    print(cmp.to_string())
    print(f"\nTOTAL: {len(final):,} noticias (meta {int(metas.meta.sum()):,})")
    print("\n== Participación (%) de cada diario dentro de su idioma ==")
    print((final.groupby("idioma").diario.value_counts(normalize=True) * 100).round(1).to_string())
    print(f"\nDataset: {(SALIDA / 'dataset_noticias.parquet').resolve()}")


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Recolección completa de noticias de IA")
    sub = ap.add_subparsers(dest="cmd", required=True)
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--diarios", default="diarios.csv")
    comun.add_argument("--metas", default="metas.csv")
    comun.add_argument("--semilla", type=int, default=42)

    u = sub.add_parser("urls", parents=[comun], help="Paso 1: URLs mensuales desde GDELT")
    u.add_argument("--proyecto", required=True, help="ID de tu proyecto de Google Cloud")
    u.add_argument("--desde", default="2021-01")
    u.add_argument("--hasta", default="2026-10")
    u.add_argument("--control", type=int, default=15, help="URLs de control por generalista y mes")
    u.add_argument("--max-gb", type=float, default=300, help="Pide confirmación si el total estimado lo supera")
    u.add_argument("--si", action="store_true", help="No pedir confirmación")
    u.set_defaults(fn=cmd_urls)

    d = sub.add_parser("descargar", parents=[comun], help="Paso 2: descarga hasta cumplir las metas")
    d.add_argument("--idiomas", default="", help="Ej. es,en (vacío = todos)")
    d.add_argument("--anios", default="", help="Ej. 2021,2022 (vacío = todos)")
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--delay", type=float, default=1.0, help="Segundos mínimos entre peticiones al mismo host")
    d.add_argument("--lote", type=int, default=60,
                   help="Descargas por lote; si interrumpes con Ctrl+C se pierde el lote en curso")
    d.add_argument("--tope", type=float, default=0.30, help="Máx. fracción de la meta por diario")
    d.add_argument("--peso-tech", type=float, default=0.15, dest="peso_tech",
                   help="Peso total de las URLs sin palabra clave de cada diario de tecnología, "
                        "relativo a sus URLs con palabra clave (0.15 = ~13%% de las descargas)")
    d.add_argument("--peso-control", type=float, default=0.05, dest="peso_control",
                   help="Igual, para la muestra de control de los generalistas")
    d.add_argument("--max-intentos", type=float, default=5.0, dest="max_intentos",
                   help="Tope de descargas por idioma-año = este factor x la meta")
    d.add_argument("--min-palabras", type=int, default=150, dest="min_palabras")
    d.add_argument("--min-menciones", type=int, default=3, dest="min_menciones")
    d.add_argument("--sin-wayback", action="store_true", dest="sin_wayback")
    d.set_defaults(fn=cmd_descargar)

    c = sub.add_parser("consolidar", parents=[comun], help="Paso 3: dataset final e informe")
    c.add_argument("--tope", type=float, default=0.30)
    c.add_argument("--sin-huella", action="store_true", dest="sin_huella",
                   help="No eliminar noticias que comparten el inicio del texto")
    c.add_argument("--huella-desde", type=int, default=0, dest="huella_desde",
                   help="Carácter desde el que se compara el texto para detectar duplicados (0 = inicio)")
    c.add_argument("--cap-estricto", action="store_true", dest="cap_estricto",
                   help="Respeta el tope por diario aunque el idioma-año quede por debajo de su meta")
    c.set_defaults(fn=cmd_consolidar)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()