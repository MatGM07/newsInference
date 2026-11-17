#!/usr/bin/env python3
"""
Mapa de cobertura de GDELT: cuántas URLs hay por diario y mes (y cuántas parecen de IA
por la URL), de 2021-01 a 2026-10, en UNA sola consulta a BigQuery.

Sirve para decidir con datos qué diarios mantener, cuáles reemplazar y cuántas noticias
de IA se pueden esperar por año. Prueba los diarios de diarios.csv y, si existe,
los de candidatos.csv (reemplazos posibles; algunos pueden no aparecer en GDELT).

Uso:
  python cobertura_gdelt.py --proyecto TU_PROYECTO_GCP

Salidas (carpeta salida/):
  cobertura_mensual.csv   una fila por diario y mes
  cobertura_resumen.csv   una fila por diario (la tabla que se imprime)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from piloto_gdelt import URL_PATRON

SQL = """
SELECT
  SourceCommonName AS dominio,
  DIV(DATE, 100000000) AS ym,
  COUNT(DISTINCT DocumentIdentifier) AS urls,
  COUNT(DISTINCT IF(REGEXP_CONTAINS(LOWER(DocumentIdentifier), @patron),
                    DocumentIdentifier, NULL)) AS urls_match
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP(@ini) AND _PARTITIONTIME < TIMESTAMP(@fin)
  AND SourceCommonName IN UNNEST(@dominios)
GROUP BY dominio, ym
"""


def resumir(mensual: pd.DataFrame, diarios: pd.DataFrame, meses: list, min_urls: int) -> pd.DataFrame:
    """mensual: diario, ym, urls, urls_match (ya agrupado por diario)."""
    filas = []
    for d in diarios.itertuples():
        m = (mensual[mensual.diario == d.diario].set_index("ym")[["urls", "urls_match"]]
             .reindex(meses, fill_value=0))
        fila = {"idioma": d.idioma, "diario": d.diario, "origen": d.origen, "tipo": d.tipo,
                "meses_cubiertos": int((m.urls >= min_urls).sum()),
                "meses_total": len(meses),
                "mediana_urls_mes": int(m.urls.median())}
        for anio in sorted({x // 100 for x in meses}):
            sel = m[[x // 100 == anio for x in m.index]]
            fila[f"match_mes_{anio}"] = round(sel.urls_match.mean(), 1)
        filas.append(fila)
    out = pd.DataFrame(filas)
    return out.sort_values(["idioma", "origen", "meses_cubiertos"],
                           ascending=[True, True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Cobertura mensual de GDELT por diario")
    ap.add_argument("--proyecto", required=True)
    ap.add_argument("--diarios", default="diarios.csv")
    ap.add_argument("--candidatos", default="candidatos.csv")
    ap.add_argument("--ini", default="2021-01", help="Primer mes AAAA-MM")
    ap.add_argument("--fin", default="2026-10", help="Último mes AAAA-MM (inclusive)")
    ap.add_argument("--min-urls", type=int, default=100,
                    help="URLs mínimas en un mes para contarlo como cubierto")
    ap.add_argument("--max-gb", type=float, default=400)
    ap.add_argument("--si", action="store_true", help="No pedir confirmación")
    args = ap.parse_args()

    base = pd.read_csv(args.diarios).assign(origen="actual")
    partes = [base]
    if Path(args.candidatos).exists():
        partes.append(pd.read_csv(args.candidatos).assign(origen="candidato"))
    diarios = pd.concat(partes, ignore_index=True).drop_duplicates("diario")
    alias = {dom.strip(): r.diario for r in diarios.itertuples()
             for dom in r.dominios_gdelt.split(";")}

    ini = pd.Timestamp(f"{args.ini}-01")
    fin = pd.Timestamp(f"{args.fin}-01") + pd.offsets.MonthBegin(1)
    meses = [p.year * 100 + p.month for p in pd.period_range(args.ini, args.fin, freq="M")]

    params = [
        bigquery.ScalarQueryParameter("patron", "STRING", URL_PATRON),
        bigquery.ScalarQueryParameter("ini", "STRING", ini.strftime("%Y-%m-%d")),
        bigquery.ScalarQueryParameter("fin", "STRING", fin.strftime("%Y-%m-%d")),
        bigquery.ArrayQueryParameter("dominios", "STRING", sorted(alias)),
    ]
    client = bigquery.Client(project=args.proyecto)
    seco = client.query(SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False))
    gb = seco.total_bytes_processed / 1e9
    print(f"La consulta procesaría ~{gb:.0f} GB (capa gratuita: 1 TB/mes, compartida con el piloto).")
    if gb > args.max_gb and not args.si:
        if input("Es más de lo esperado. ¿Continuar? [s/N] ").strip().lower() != "s":
            sys.exit("Cancelado.")

    df = client.query(SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=params)).to_dataframe()
    if df.empty:
        sys.exit("GDELT no devolvió datos. Revisa los nombres de dominio.")
    df["diario"] = df.dominio.map(alias)
    mensual = df.groupby(["diario", "ym"], as_index=False)[["urls", "urls_match"]].sum()

    out = Path("salida")
    out.mkdir(exist_ok=True)
    mensual.to_csv(out / "cobertura_mensual.csv", index=False)
    resumen = resumir(mensual, diarios, meses, args.min_urls)
    resumen.to_csv(out / "cobertura_resumen.csv", index=False)
    print("\n" + resumen.to_string(index=False))
    print(f"\nGuardado en {out.resolve()}")


if __name__ == "__main__":
    main()
