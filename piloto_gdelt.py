#!/usr/bin/env python3
"""
Piloto de recolección de noticias de IA: GDELT (URLs) + trafilatura (texto).

Qué hace, para UN mes:
  1. Consulta GDELT GKG en BigQuery: todas las URLs del mes de los diarios de diarios.csv.
     Marca cuáles tienen en la URL palabras clave de IA ("match_url").
  2. Muestrea `--max` URLs candidatas (match_url) y `--control` URLs de control (sin match)
     por diario. El control sirve para estimar cuántas noticias de IA se pierden
     al filtrar solo por la URL.
  3. Descarga cada URL (con respaldo en Wayback Machine) y extrae el texto con trafilatura.
  4. Decide si el texto trata de IA y genera un resumen por diario.

Uso:
  pip install -r requirements.txt
  gcloud auth application-default login          # o autenticarte desde Colab
  python piloto_gdelt.py --mes 2023-06 --proyecto TU_PROYECTO_GCP
  python piloto_gdelt.py --mes 2021-06 --proyecto TU_PROYECTO_GCP    # para comparar enlaces rotos

Salidas (carpeta salida/):
  gdelt_urls_<mes>.parquet        URLs de GDELT (caché, no se vuelve a consultar)
  piloto_<mes>_articulos.parquet  artículos descargados
  piloto_<mes>_resumen.csv        métricas por diario
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from google.cloud import bigquery
from tqdm import tqdm
from trafilatura.settings import use_config

# ---------------------------------------------------------------- filtros de IA
# Palabras clave dentro de la URL (sintaxis RE2 de BigQuery). Recall parcial a propósito:
# el control mide cuánto se escapa.
URL_PATRON = (
    r"artificial[-_]intelligence|inteligencia[-_]artificial|intelig[eê]ncia[-_]artificial|"
    r"intelligence[-_]artificielle|k(?:ü|u|ue|%c3%bc)nstliche[-_]intelligenz|chatgpt|openai|"
    r"gpt[-_]?[0-9]|deepmind|generative[-_]ai|ia[-_]generativa|llm|machine[-_]learning|"
    r"aprendizaje[-_]autom|deepfake|chatbot|[-_/]ai[-_/.]|[-_/]ia[-_/.]|[-_/]ki[-_/.]"
)

# Términos largos (insensible a mayúsculas) y siglas (sensible a mayúsculas) para el texto.
IA_LARGO = re.compile(
    r"artificial intelligence|inteligencia artificial|intelig[eê]ncia artificial|"
    r"intelligence artificielle|k[üu]nstliche intelligenz|chatgpt|openai|generative ai|"
    r"ia generativa|machine learning|aprendizaje autom[aá]tico|aprendizado de m[aá]quina|"
    r"apprentissage automatique|maschinelles lernen|large language model|gpt-\d|"
    r"deepmind|chatbot|deepfake",
    re.I,
)
IA_SIGLAS = re.compile(r"\b(?:AI|A\.I\.|IA|KI)\b")


def menciones_ia(texto: str) -> int:
    return len(IA_LARGO.findall(texto)) + len(IA_SIGLAS.findall(texto))


def clasificar_ia(titulo: str, texto: str):
    """Devuelve (n_menciones, es_ia). Es IA si el título la menciona o hay >= 3 menciones."""
    n = menciones_ia(texto or "")
    en_titulo = menciones_ia(titulo or "") > 0
    return n, bool(en_titulo or n >= 3)


# ---------------------------------------------------------------- BigQuery
SQL = """
SELECT
  dominio, url, fecha,
  REGEXP_CONTAINS(LOWER(url), @patron) AS match_url
FROM (
  SELECT DISTINCT
    SourceCommonName AS dominio,
    DocumentIdentifier AS url,
    SUBSTR(CAST(DATE AS STRING), 1, 8) AS fecha
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP(@ini) AND _PARTITIONTIME < TIMESTAMP(@fin)
    AND DATE >= @dini AND DATE < @dfin
    AND SourceCommonName IN UNNEST(@dominios)
)
"""


def consultar_gdelt(args, dominios, ruta_cache: Path) -> pd.DataFrame:
    if ruta_cache.exists():
        print(f"Usando caché de GDELT: {ruta_cache}")
        return pd.read_parquet(ruta_cache)

    ini = pd.Timestamp(f"{args.mes}-01")
    fin = ini + pd.offsets.MonthBegin(1)
    params = [
        bigquery.ScalarQueryParameter("patron", "STRING", URL_PATRON),
        bigquery.ScalarQueryParameter("ini", "STRING", ini.strftime("%Y-%m-%d")),
        bigquery.ScalarQueryParameter("fin", "STRING", fin.strftime("%Y-%m-%d")),
        bigquery.ScalarQueryParameter("dini", "INT64", int(ini.strftime("%Y%m%d") + "000000")),
        bigquery.ScalarQueryParameter("dfin", "INT64", int(fin.strftime("%Y%m%d") + "000000")),
        bigquery.ArrayQueryParameter("dominios", "STRING", sorted(dominios)),
    ]
    client = bigquery.Client(project=args.proyecto)

    seco = client.query(SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False))
    gb = seco.total_bytes_processed / 1e9
    print(f"La consulta procesaría ~{gb:.1f} GB (la capa gratuita es 1 TB/mes).")
    if gb > args.max_gb and not args.si:
        if input("Es más de lo esperado. ¿Continuar? [s/N] ").strip().lower() != "s":
            sys.exit("Cancelado.")

    df = client.query(SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=params)).to_dataframe()
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta_cache, index=False)
    return df


# ---------------------------------------------------------------- descarga
CONFIG = use_config()
CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "20")
CONFIG.set("DEFAULT", "MAX_REDIRECTS", "5")

_glock = threading.Lock()
_locks: dict = {}
_ultimo: dict = {}


def esperar(host: str, delay: float):
    """Espacia las peticiones al mismo host para no saturarlo."""
    with _glock:
        lk = _locks.setdefault(host, threading.Lock())
    with lk:
        dt = time.time() - _ultimo.get(host, 0.0)
        if dt < delay:
            time.sleep(delay - dt)
        _ultimo[host] = time.time()


def desde_wayback(url: str, fecha: str):
    try:
        esperar("web.archive.org", 1.0)
        r = requests.get("https://archive.org/wayback/available",
                         params={"url": url, "timestamp": fecha}, timeout=15)
        snap = (r.json().get("archived_snapshots") or {}).get("closest")
        if not snap or not snap.get("available"):
            return None
        # "id_" pide la página original sin la barra de Wayback
        snap_url = re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap["url"], count=1)
        esperar("web.archive.org", 1.0)
        return trafilatura.fetch_url(snap_url, config=CONFIG)
    except Exception:
        return None


def descargar(fila: dict, usar_wayback: bool, delay: float) -> dict:
    url = fila["url"]
    res = {**fila, "via": None, "titulo": None, "fecha_pub": None, "texto": None,
           "n_palabras": 0, "n_menciones_ia": 0, "es_ia": False, "error": None}
    try:
        esperar(urlparse(url).netloc, delay)
        html, via = trafilatura.fetch_url(url, config=CONFIG), "directo"
        if not html and usar_wayback:
            html, via = desde_wayback(url, fila["fecha"]), "wayback"
        if not html:
            res["error"] = "sin_descarga"
            return res
        salida = trafilatura.extract(html, output_format="json", include_comments=False,
                                     include_tables=False, favor_precision=True, config=CONFIG)
        if not salida:
            res["error"] = "sin_texto"
            return res
        d = json.loads(salida)
        texto = d.get("text") or ""
        n, es_ia = clasificar_ia(d.get("title") or "", texto)
        res.update(via=via, titulo=d.get("title"), fecha_pub=d.get("date"), texto=texto,
                   n_palabras=len(texto.split()), n_menciones_ia=n, es_ia=es_ia)
    except Exception as e:  # un fallo no debe tumbar el piloto
        res["error"] = f"excepcion:{type(e).__name__}"
    return res


# ---------------------------------------------------------------- resumen
def tasa(a, b):
    return round(a / b, 3) if b else 0.0


def construir_resumen(diarios: pd.DataFrame, urls: pd.DataFrame, arts: pd.DataFrame,
                      min_palabras: int) -> pd.DataFrame:
    filas = []
    for _, d in diarios.iterrows():
        k = d["diario"]
        u = urls[urls.diario == k]
        total, match = len(u), int(u.match_url.sum())
        fila = {"idioma": d["idioma"], "diario": k, "tipo": d["tipo"],
                "acceso_estimado": d["acceso_estimado"],
                "urls_gdelt_mes": total, "urls_con_match": match}
        est_ia = 0.0
        for tipo, sufijo, universo in (("candidata", "cand", match), ("control", "ctrl", total - match)):
            a = arts[(arts.diario == k) & (arts.tipo == tipo)] if len(arts) else arts
            n = len(a)
            con_texto = int((a.n_palabras >= min_palabras).sum()) if n else 0
            ia = int(((a.n_palabras >= min_palabras) & a.es_ia).sum()) if n else 0
            via_wb = int((a.via == "wayback").sum()) if n else 0
            fila.update({f"intentadas_{sufijo}": n,
                         f"texto_ok_{sufijo}": con_texto,
                         f"wayback_{sufijo}": via_wb,
                         f"es_ia_{sufijo}": ia,
                         f"tasa_texto_ok_{sufijo}": tasa(con_texto, n),
                         f"tasa_ia_{sufijo}": tasa(ia, con_texto)})
            est_ia += universo * tasa(ia, con_texto)
        # IA estimada que existe en el mes (con texto accesible) y cuánto capta el filtro por URL
        est_match = match * fila["tasa_ia_cand"]
        fila["ia_estimadas_mes"] = round(est_ia)
        fila["recall_filtro_url"] = tasa(est_match, est_ia)
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Piloto GDELT + trafilatura")
    ap.add_argument("--mes", required=True, help="Mes a probar, formato AAAA-MM (ej. 2023-06)")
    ap.add_argument("--proyecto", required=True, help="ID de tu proyecto de Google Cloud")
    ap.add_argument("--diarios", default="diarios.csv")
    ap.add_argument("--idiomas", default="", help="Filtrar idiomas, ej. es,en (vacío = todos)")
    ap.add_argument("--max", type=int, default=40, help="URLs candidatas a descargar por diario")
    ap.add_argument("--control", type=int, default=20, help="URLs de control por diario")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.0, help="Segundos mínimos entre peticiones al mismo host")
    ap.add_argument("--min-palabras", type=int, default=150, help="Mínimo para considerar el texto completo")
    ap.add_argument("--sin-wayback", action="store_true")
    ap.add_argument("--max-gb", type=float, default=100, help="Pide confirmación si la consulta supera esto")
    ap.add_argument("--si", action="store_true", help="No pedir confirmación")
    args = ap.parse_args()

    diarios = pd.read_csv(args.diarios)
    if args.idiomas:
        diarios = diarios[diarios.idioma.isin(args.idiomas.split(","))]
    alias = {dom.strip(): r.diario for r in diarios.itertuples()
             for dom in r.dominios_gdelt.split(";")}

    out = Path("salida")
    urls = consultar_gdelt(args, alias.keys(), out / f"gdelt_urls_{args.mes}.parquet")
    if urls.empty:
        sys.exit("GDELT no devolvió URLs. Revisa los nombres de SourceCommonName en diarios.csv.")
    urls["diario"] = urls.dominio.map(alias)
    urls = urls.drop_duplicates("url")
    sin = sorted(set(diarios.diario) - set(urls.diario))
    if sin:
        print("Diarios sin URLs en GDELT (revisar el nombre en dominios_gdelt):", ", ".join(sin))

    mezcla = urls.sample(frac=1, random_state=42)
    cand = mezcla[mezcla.match_url].groupby("diario").head(args.max).assign(tipo="candidata")
    ctrl = mezcla[~mezcla.match_url].groupby("diario").head(args.control).assign(tipo="control")
    muestra = pd.concat([cand, ctrl], ignore_index=True)
    idioma_de = dict(zip(diarios.diario, diarios.idioma))
    muestra["idioma"] = muestra.diario.map(idioma_de)
    print(f"\nURLs en GDELT: {len(urls):,} | a descargar: {len(muestra):,} "
          f"({len(cand):,} candidatas + {len(ctrl):,} control)\n")

    filas = muestra.to_dict("records")
    resultados = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(descargar, f, not args.sin_wayback, args.delay) for f in filas]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Descargando"):
            resultados.append(fut.result())
    arts = pd.DataFrame(resultados)

    out.mkdir(exist_ok=True)
    arts.to_parquet(out / f"piloto_{args.mes}_articulos.parquet", index=False)
    resumen = construir_resumen(diarios, urls, arts, args.min_palabras)
    resumen.to_csv(out / f"piloto_{args.mes}_resumen.csv", index=False)

    cols = ["idioma", "diario", "urls_gdelt_mes", "urls_con_match", "intentadas_cand",
            "tasa_texto_ok_cand", "wayback_cand", "tasa_ia_cand", "tasa_ia_ctrl",
            "ia_estimadas_mes", "recall_filtro_url"]
    print("\n" + resumen[cols].to_string(index=False))
    print(f"\nGuardado en {out.resolve()}")


if __name__ == "__main__":
    main()
