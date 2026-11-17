#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ejecutar_modelo.py  -  Análisis de emociones con Horizon-Labs/multilingual-emotions-base.

RESPONSABILIDAD ÚNICA: leer datos/preparados/noticias_preparadas.parquet, ejecutar el
modelo y guardar los resultados en resultados/. NO lee los archivos originales, NO limpia
textos y NO llama a preparar_datos.py. Si el archivo preparado no existe, se detiene y te
dice que ejecutes primero preparar_datos.py.

Qué es el modelo (según su config.json y su model card): ModernBERT multilingüe (mmBERT-base),
clasificación MULTIETIQUETA con salida sigmoide independiente para 28 etiquetas de GoEmotions
(incluida 'neutral'). Se entrenó con textos de hasta 256 tokens, por eso las noticias se
dividen en fragmentos de como máximo 256 tokens.

Uso típico:
    python ejecutar_modelo.py --muestra 100          # prueba: 100 noticias por idioma
    python ejecutar_modelo.py --solo-contar          # cuenta fragmentos sin cargar el modelo
    python ejecutar_modelo.py                        # corpus completo (reanuda si se interrumpe)
    python ejecutar_modelo.py --umbral 0.4           # otro umbral SIN volver a correr el modelo
    python ejecutar_modelo.py --forzar               # recalcula todo desde cero

Cómo se agrega por noticia (los fragmentos NO se promedian "a ojo"):
  * Cada fragmento pesa según su número de tokens (un fragmento de 5 tokens pesa mucho menos).
  * media_<emoción> : probabilidad media ponderada por tokens.
  * max_<emoción>   : probabilidad máxima entre los fragmentos.
  * prev_<emoción>  : prevalencia = fracción de tokens (de la noticia) en fragmentos donde la
                      probabilidad supera el umbral de esa emoción.
  * Una emoción queda DETECTADA en la noticia si prevalencia >= --min-prevalencia (def. 0.5).
    Además se guarda 'emociones_en_algun_fragmento' (basta un fragmento sobre el umbral).
  * Umbral por defecto: el thresholds.json oficial del modelo (uno por emoción).
    Con --umbral X se usa X para todas.
  * Se añaden las 6 emociones de Ekman con el ekman_mapping.json oficial (máximo por grupo,
    como indica el model card), agregadas con la misma lógica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --------------------------------------------------------------------------
# Rutas (relativas a la ubicación de este archivo)
# --------------------------------------------------------------------------
PROYECTO = Path(__file__).resolve().parent
ARCHIVO_PREPARADO = PROYECTO / "datos" / "preparados" / "noticias_preparadas.parquet"
DIR_RESULTADOS = PROYECTO / "resultados"
DIR_MODELOS = PROYECTO / "modelos"
MODELO_ID = "Horizon-Labs/multilingual-emotions-base"
ALGORITMO = "v1-parrafos-oraciones"   # cambia si cambia la forma de fragmentar
COL_TEXTO = "texto_limpio"
COLUMNAS_TEXTO_PESADAS = ("texto_original", "texto_limpio")
RE_ORACION = re.compile(r"(?<=[.!?…»”\"])\s+")

torch = None  # se importa solo cuando hace falta (no se necesita para --solo-contar)


def barra(**kw):
    return tqdm(**kw) if tqdm else None


def hash_texto(t: str) -> str:
    return hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def escritura_atomica_parquet(tabla: pa.Table, destino: Path):
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    pq.write_table(tabla, tmp, compression="zstd")
    os.replace(tmp, destino)


# --------------------------------------------------------------------------
# Carga del modelo / tokenizer (se descarga UNA vez y se reutiliza de la caché local)
# --------------------------------------------------------------------------
def con_cache_local(fn, *a, solo_local: bool, **kw):
    """Primero intenta la caché local (sin red); si no está, descarga (a menos que --sin-descarga)."""
    try:
        return fn(*a, local_files_only=True, **kw)
    except Exception:
        if solo_local:
            raise
        print("   (no está en la caché local: descargando desde Hugging Face; solo ocurre la primera vez)")
        return fn(*a, local_files_only=False, **kw)


def archivo_del_modelo(modelo: str, nombre: str, dir_modelos: Path, solo_local: bool) -> Path:
    from huggingface_hub import hf_hub_download
    if Path(modelo).is_dir():
        return Path(modelo) / nombre
    return Path(con_cache_local(hf_hub_download, modelo, nombre, solo_local=solo_local, cache_dir=str(dir_modelos)))


def cargar_tokenizador(modelo: str, dir_modelos: Path, solo_local: bool):
    from transformers import AutoTokenizer
    tok = con_cache_local(AutoTokenizer.from_pretrained, modelo, solo_local=solo_local, cache_dir=str(dir_modelos))
    tok.model_max_length = int(1e9)  # evita avisos: nosotros controlamos la longitud
    return tok


def cargar_modelo(modelo: str, dir_modelos: Path, solo_local: bool, dispositivo: str, hilos: int | None):
    global torch
    import torch as _torch
    torch = _torch
    from transformers import AutoModelForSequenceClassification
    if hilos:
        torch.set_num_threads(hilos)
    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Pediste --dispositivo cuda, pero este PyTorch no detecta una GPU NVIDIA.")
    print(f"Cargando modelo '{modelo}' en {dispositivo.upper()} (torch {torch.__version__}, hilos CPU: {torch.get_num_threads()}) ...")
    m = con_cache_local(AutoModelForSequenceClassification.from_pretrained, modelo, solo_local=solo_local,
                        cache_dir=str(dir_modelos))
    m.eval()
    m.to(dispositivo)
    return m, dispositivo


def datos_del_modelo(modelo_cfg, modelo: str, dir_modelos: Path, solo_local: bool, umbral_global):
    """Etiquetas, umbrales y mapeo Ekman: todo leído de los archivos oficiales, nada inventado."""
    cfg = modelo_cfg
    if getattr(cfg, "problem_type", None) != "multi_label_classification":
        raise SystemExit(f"El modelo no declara 'multi_label_classification' (declara: {cfg.problem_type}). "
                         "Este pipeline asume salida multietiqueta; revisa el model card.")
    etiquetas = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    p_thr = archivo_del_modelo(modelo, "thresholds.json", dir_modelos, solo_local)
    p_ekm = archivo_del_modelo(modelo, "ekman_mapping.json", dir_modelos, solo_local)
    thr_oficial = json.loads(p_thr.read_text(encoding="utf-8"))
    ekman = json.loads(p_ekm.read_text(encoding="utf-8"))
    faltan = [e for e in etiquetas if e not in thr_oficial]
    if faltan:
        print(f"ADVERTENCIA: thresholds.json no trae umbral para {faltan}; se usa 0.5 en ellas.")
    if umbral_global is not None:
        thr = np.full(len(etiquetas), float(umbral_global), dtype=np.float64)
        modo = f"global={umbral_global}"
    else:
        thr = np.array([float(thr_oficial.get(e, 0.5)) for e in etiquetas], dtype=np.float64)
        modo = "por_emocion(thresholds.json oficial)"
    for g, ls in ekman.items():
        malas = [l for l in ls if l not in etiquetas]
        if malas:
            raise SystemExit(f"ekman_mapping.json menciona etiquetas inexistentes en el modelo: {malas}")
    partes = p_thr.parent.parts
    revision = partes[partes.index("snapshots") + 1] if "snapshots" in partes else "local"
    return etiquetas, thr, modo, ekman, revision


# --------------------------------------------------------------------------
# Fragmentación por tokens (respeta el límite; no corta frases si se puede evitar)
# --------------------------------------------------------------------------
def tokens_especiales(tok) -> tuple[list[int], list[int]]:
    """Descubre (sin suponer) qué tokens especiales añade el tokenizer antes/después del texto."""
    con = tok("a", add_special_tokens=True)["input_ids"]
    sin = tok("a", add_special_tokens=False)["input_ids"]
    for i in range(len(con) - len(sin) + 1):
        if con[i:i + len(sin)] == sin:
            return con[:i], con[i + len(sin):]
    raise RuntimeError("No pude determinar los tokens especiales del tokenizer.")


def tokenizar(tok, textos: list[str]) -> list[list[int]]:
    if not textos:
        return []
    return tok(textos, add_special_tokens=False, return_attention_mask=False)["input_ids"]


def dividir_parrafo_largo(texto: str, presupuesto: int, tok, guardar_texto: bool):
    """Un párrafo que no cabe: se parte por oraciones; una oración gigante se parte por ventanas de tokens."""
    oraciones = [s for s in RE_ORACION.split(texto) if s.strip()]
    ids_or = tokenizar(tok, oraciones)
    piezas, cur_ids, cur_t = [], [], []

    def volcar():
        nonlocal cur_ids, cur_t
        if cur_ids:
            piezas.append((" ".join(cur_t), cur_ids))
        cur_ids, cur_t = [], []

    for s, ids in zip(oraciones, ids_or):
        if not ids:
            continue
        if len(ids) > presupuesto:
            volcar()
            for k in range(0, len(ids), presupuesto):
                w = ids[k:k + presupuesto]
                piezas.append((tok.decode(w) if guardar_texto else "", w))
            continue
        if cur_ids and len(cur_ids) + len(ids) > presupuesto:
            volcar()
        cur_ids = cur_ids + ids
        cur_t.append(s)
    volcar()
    return piezas


def fragmentar(parrafos: list[str], ids_parr: list[list[int]], presupuesto: int, tok, guardar_texto: bool):
    """Empaqueta párrafos consecutivos en fragmentos de <= presupuesto tokens (sin superposición)."""
    piezas = []
    for t, ids in zip(parrafos, ids_parr):
        if not ids:
            continue
        if len(ids) <= presupuesto:
            piezas.append((t, ids))
        else:
            piezas.extend(dividir_parrafo_largo(t, presupuesto, tok, guardar_texto))
    frags, cur_ids, cur_t = [], [], []
    for t, ids in piezas:
        if cur_ids and len(cur_ids) + len(ids) > presupuesto:
            frags.append((cur_ids, "\n".join(cur_t) if guardar_texto else None))
            cur_ids, cur_t = [], []
        cur_ids = cur_ids + ids
        cur_t.append(t)
    if cur_ids:
        frags.append((cur_ids, "\n".join(cur_t) if guardar_texto else None))
    return frags


def fragmentar_bloque(textos: list[str], presupuesto: int, tok, max_fragmentos: int, guardar_texto: bool):
    """Devuelve, por noticia, (n_total_fragmentos, [(indice_original, ids, texto|None), ...]) ya con tope aplicado."""
    por_noticia_parrafos = [[p for p in t.split("\n") if p.strip()] for t in textos]
    planos = [p for ps in por_noticia_parrafos for p in ps]
    ids_planos = tokenizar(tok, planos)
    salida, pos = [], 0
    for ps in por_noticia_parrafos:
        ids_ps = ids_planos[pos:pos + len(ps)]
        pos += len(ps)
        frs = fragmentar(ps, ids_ps, presupuesto, tok, guardar_texto)
        total = len(frs)
        indices = list(range(total))
        if max_fragmentos and total > max_fragmentos:  # muestreo uniforme: cubre todo el texto
            indices = sorted(set(np.linspace(0, total - 1, max_fragmentos).round().astype(int).tolist()))
        salida.append((total, [(i, frs[i][0], frs[i][1]) for i in indices]))
    return salida


# --------------------------------------------------------------------------
# Inferencia por lotes con control de memoria y errores
# --------------------------------------------------------------------------
def predecir(modelo, dispositivo: str, lista_ids: list[list[int]], pre: list[int], suf: list[int],
             pad_id: int, lote: int, n_etq: int):
    """Probabilidades sigmoide [n, n_etq]. Ordena por longitud (menos relleno). Si un lote falla
    (p. ej. memoria), lo reintenta con la mitad; si un fragmento solo falla, devuelve NaN y su error."""
    n = len(lista_ids)
    out = np.full((n, n_etq), np.nan, dtype=np.float32)
    errores: dict[int, str] = {}
    orden = sorted(range(n), key=lambda i: -len(lista_ids[i]))
    i, tam = 0, max(1, lote)
    while i < n:
        idx = orden[i:i + tam]
        try:
            secuencias = [pre + lista_ids[j] + suf for j in idx]
            L = max(len(s) for s in secuencias)
            ids_t = torch.full((len(idx), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, s in enumerate(secuencias):
                ids_t[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[r, :len(s)] = 1
            with torch.inference_mode():
                logits = modelo(input_ids=ids_t.to(dispositivo), attention_mask=mask.to(dispositivo)).logits
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
            i += len(idx)
        except Exception as e:  # noqa: BLE001  (queremos sobrevivir a cualquier fallo de un lote)
            if dispositivo == "cuda":
                torch.cuda.empty_cache()
            if tam > 1:
                tam = max(1, tam // 2)
                print(f"\n   Aviso: falló un lote ({type(e).__name__}); reintento con lote={tam}")
                continue
            errores[idx[0]] = f"{type(e).__name__}: {e}"
            i += 1
    return out, errores


# --------------------------------------------------------------------------
# Lectura del archivo preparado
# --------------------------------------------------------------------------
def leer_metadatos(ruta: Path) -> pd.DataFrame:
    nombres = pq.ParquetFile(ruta).schema_arrow.names
    if "id" not in nombres or COL_TEXTO not in nombres:
        raise SystemExit(f"El archivo preparado no tiene las columnas 'id' y '{COL_TEXTO}'. "
                         "Vuelve a ejecutar preparar_datos.py.")
    cols = [c for c in nombres if c not in COLUMNAS_TEXTO_PESADAS]
    return pq.read_table(ruta, columns=cols).to_pandas()


def iterar_textos(ruta: Path, tam=2000):
    pf = pq.ParquetFile(ruta)
    for b in pf.iter_batches(batch_size=tam, columns=["id", COL_TEXTO]):
        d = b.to_pydict()
        yield d["id"], d[COL_TEXTO]


# --------------------------------------------------------------------------
# Agregación fragmentos -> noticia
# --------------------------------------------------------------------------
def agregar(frag: pd.DataFrame, etiquetas, thr, ekman, min_prev):
    """frag: una fila por fragmento con n_tokens y columnas p_<etiqueta>. Devuelve DataFrame por noticia."""
    L = len(etiquetas)
    P = frag[[f"p_{e}" for e in etiquetas]].to_numpy(np.float64)
    w = frag["n_tokens"].to_numpy(np.float64)
    codigos, ids = pd.factorize(frag["id"])
    G = len(ids)
    sw = np.bincount(codigos, weights=w, minlength=G)
    sw_seguro = np.where(sw > 0, sw, np.nan)
    media = np.empty((G, L))
    prev = np.empty((G, L))
    for j in range(L):
        media[:, j] = np.bincount(codigos, weights=w * P[:, j], minlength=G) / sw_seguro
        prev[:, j] = np.bincount(codigos, weights=w * (P[:, j] >= thr[j]), minlength=G) / sw_seguro
    mx = pd.DataFrame(P).groupby(codigos).max().to_numpy()
    det = prev >= min_prev
    alguno = mx >= thr[None, :]

    grupos = list(ekman.keys())
    E = np.column_stack([P[:, [etiquetas.index(l) for l in ekman[g]]].max(axis=1) for g in grupos])
    e_media = np.empty((G, len(grupos)))
    for j in range(len(grupos)):
        e_media[:, j] = np.bincount(codigos, weights=w * E[:, j], minlength=G) / sw_seguro
    e_max = pd.DataFrame(E).groupby(codigos).max().to_numpy()
    e_det = np.column_stack([det[:, [etiquetas.index(l) for l in ekman[g]]].any(axis=1) for g in grupos])

    et = np.array(etiquetas)
    df = pd.DataFrame({"id": ids})
    df["n_fragmentos_procesados"] = np.bincount(codigos, minlength=G).astype(np.int32)
    df["n_tokens_total"] = sw.astype(np.int64)
    df["emociones_detectadas"] = [";".join(et[r]) for r in det]
    df["n_emociones_detectadas"] = det.sum(axis=1).astype(np.int16)
    df["emociones_en_algun_fragmento"] = [";".join(et[r]) for r in alguno]
    df["ekman_detectadas"] = [";".join(np.array(grupos)[r]) for r in e_det]
    cols = {}
    for j, e in enumerate(etiquetas):
        cols[f"det_{e}"] = det[:, j]
    for j, e in enumerate(etiquetas):
        cols[f"media_{e}"] = media[:, j].astype(np.float32)
    for j, e in enumerate(etiquetas):
        cols[f"max_{e}"] = mx[:, j].astype(np.float32)
    for j, e in enumerate(etiquetas):
        cols[f"prev_{e}"] = prev[:, j].astype(np.float32)
    for j, g in enumerate(grupos):
        cols[f"ekman_media_{g}"] = e_media[:, j].astype(np.float32)
    for j, g in enumerate(grupos):
        cols[f"ekman_max_{g}"] = e_max[:, j].astype(np.float32)
    return pd.concat([df, pd.DataFrame(cols)], axis=1)


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Ejecuta el modelo de emociones sobre noticias_preparadas.parquet.")
    ap.add_argument("--modelo", default=MODELO_ID, help="Id en Hugging Face o carpeta local con el modelo.")
    ap.add_argument("--dir-modelos", type=Path, default=DIR_MODELOS, help="Caché local del modelo (se descarga una vez).")
    ap.add_argument("--sin-descarga", action="store_true", help="Prohíbe descargar: solo usa la caché local.")
    ap.add_argument("--lote", type=int, default=16, help="Tamaño de lote (fragmentos por pasada).")
    ap.add_argument("--max-tokens", type=int, default=256, help="Máx. tokens por fragmento, incl. especiales (el modelo se entrenó con 256).")
    ap.add_argument("--max-fragmentos", type=int, default=0, help="Tope de fragmentos por noticia (0 = sin tope; si hay tope, muestreo uniforme).")
    ap.add_argument("--umbral", type=float, default=None, help="Umbral global (por defecto: thresholds.json oficial, uno por emoción).")
    ap.add_argument("--min-prevalencia", type=float, default=0.5, help="Fracción mínima de tokens sobre el umbral para dar la emoción por detectada en la noticia.")
    ap.add_argument("--muestra", type=int, default=None, help="Procesa N noticias por idioma (salida en resultados/prueba/).")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--idiomas", nargs="+", default=None, help="Limita a estos idiomas, p. ej. es en.")
    ap.add_argument("--forzar", action="store_true", help="Ignora resultados previos y recalcula todo.")
    ap.add_argument("--solo-contar", action="store_true", help="Solo cuenta fragmentos (no carga el modelo, solo el tokenizer).")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--hilos", type=int, default=None, help="Hilos de CPU para PyTorch (por defecto: todos).")
    ap.add_argument("--tamano-bloque", type=int, default=256, help="Noticias por bloque (se guarda progreso en cada bloque).")
    ap.add_argument("--guardar-texto-fragmentos", action="store_true", help="Incluye el texto de cada fragmento en el archivo por fragmento (pesa más).")
    ap.add_argument("--csv", action="store_true", help="Además del Parquet, escribe emociones_por_noticia.csv.")
    args = ap.parse_args()

    # 1) El archivo preparado debe existir (condición fundamental del proyecto)
    if not ARCHIVO_PREPARADO.exists():
        print("No encuentro el archivo preparado:\n  " + str(ARCHIVO_PREPARADO))
        print("\nPrimero debes ejecutar:  python preparar_datos.py")
        print("(ejecutar_modelo.py solo lee ese archivo; no prepara datos por su cuenta.)")
        return 1

    t0 = time.time()
    salida_dir = DIR_RESULTADOS / "prueba" if args.muestra else DIR_RESULTADOS
    salida_dir.mkdir(parents=True, exist_ok=True)
    args.dir_modelos.mkdir(parents=True, exist_ok=True)
    print(f"Entrada : {ARCHIVO_PREPARADO}")
    print(f"Salida  : {salida_dir}")

    meta = leer_metadatos(ARCHIVO_PREPARADO)
    total_corpus = len(meta)
    if args.idiomas:
        meta = meta[meta["idioma"].isin(args.idiomas)]
    if args.muestra:
        partes = [g.sample(min(args.muestra, len(g)), random_state=args.semilla) for _, g in meta.groupby("idioma")]
        meta = pd.concat(partes)
        print(f"*** MODO MUESTRA: {args.muestra} noticias por idioma ({len(meta)} en total) ***")
    meta = meta.reset_index(drop=True)
    ids_objetivo = set(meta["id"].astype(str))
    print(f"Noticias a procesar: {len(meta):,} de {total_corpus:,} en el archivo preparado")

    # 2) Tokenizer
    print("Cargando tokenizer ...")
    tok = cargar_tokenizador(args.modelo, args.dir_modelos, args.sin_descarga)
    pre, suf = tokens_especiales(tok)
    presupuesto = args.max_tokens - len(pre) - len(suf)
    if presupuesto < 8:
        raise SystemExit("--max-tokens es demasiado pequeño.")
    if args.max_tokens > 256:
        print("ADVERTENCIA: el model card solo documenta entrenamiento con 256 tokens; más de eso no está validado.")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ---- Modo solo contar -------------------------------------------------
    if args.solo_contar:
        filas = []
        pend = []
        idioma_de = dict(zip(meta["id"].astype(str), meta["idioma"]))
        def contar(bloque):
            res = fragmentar_bloque([t for _, t in bloque], presupuesto, tok, args.max_fragmentos, False)
            for (i, _), (tot, frs) in zip(bloque, res):
                filas.append({"idioma": idioma_de[i], "fragmentos": tot, "tokens": sum(len(f[1]) for f in frs)})
        barra_c = barra(total=len(ids_objetivo), desc="contando", unit="noticia")
        for ids, textos in iterar_textos(ARCHIVO_PREPARADO):
            for i, t in zip(ids, textos):
                if i in ids_objetivo:
                    pend.append((i, t))
            while len(pend) >= args.tamano_bloque:
                bl, pend = pend[:args.tamano_bloque], pend[args.tamano_bloque:]
                contar(bl)
                if barra_c: barra_c.update(len(bl))
        if pend:
            contar(pend)
            if barra_c: barra_c.update(len(pend))
        if barra_c: barra_c.close()
        d = pd.DataFrame(filas)
        print(f"\nFragmentos totales (max {args.max_tokens} tokens c/u): {d.fragmentos.sum():,} | tokens: {d.tokens.sum():,}")
        print(d.groupby("idioma").agg(noticias=("fragmentos", "size"), fragmentos=("fragmentos", "sum"),
                                      fr_por_noticia=("fragmentos", "mean"), max_fr=("fragmentos", "max")).round(2).to_string())
        print(f"Percentiles de fragmentos por noticia: {d.fragmentos.quantile([.5, .95, .99]).to_dict()}")
        return 0

    # 3) Modelo y datos del modelo
    modelo, dispositivo = cargar_modelo(args.modelo, args.dir_modelos, args.sin_descarga, args.dispositivo, args.hilos)
    etiquetas, thr, modo_umbral, ekman, revision = datos_del_modelo(
        modelo.config, args.modelo, args.dir_modelos, args.sin_descarga, args.umbral)
    n_etq = len(etiquetas)
    if args.max_tokens > getattr(modelo.config, "max_position_embeddings", 10**9):
        raise SystemExit("--max-tokens supera max_position_embeddings del modelo.")
    print(f"Modelo OK | etiquetas: {n_etq} | revisión: {revision[:12]} | umbral: {modo_umbral}")

    # 4) Caché de probabilidades por fragmento (válida para mismo modelo + misma forma de fragmentar)
    huella = hash_texto(json.dumps({"modelo": args.modelo, "revision": revision, "max_tokens": args.max_tokens,
                                    "max_fragmentos": args.max_fragmentos, "algoritmo": ALGORITMO,
                                    "pre": pre, "suf": suf, "texto_fragmentos": args.guardar_texto_fragmentos},
                                   sort_keys=True))[:16]
    cache_dir = DIR_RESULTADOS / "cache_fragmentos" / huella
    if args.forzar and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("--forzar: se descartaron los resultados previos de esta configuración.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    hechos: set[tuple[str, str]] = set()
    for p in cache_dir.glob("parte_*.parquet"):
        t = pq.read_table(p, columns=["id", "hash_texto"]).to_pandas()
        hechos.update(zip(t["id"], t["hash_texto"]))

    # 5) ¿Qué noticias faltan? (id + hash del texto limpio; no se repite lo ya calculado)
    hash_de: dict[str, str] = {}
    for ids, textos in iterar_textos(ARCHIVO_PREPARADO):
        for i, t in zip(ids, textos):
            if i in ids_objetivo:
                hash_de[i] = hash_texto(t or "")
    pendientes = [i for i in meta["id"].astype(str) if (i, hash_de.get(i)) not in hechos]
    print(f"Con resultados válidos reutilizados: {len(meta) - len(pendientes):,} | por calcular: {len(pendientes):,}")

    errores_run: dict[str, str] = {}
    n_frag_calculados = n_tok_calculados = 0
    seg_modelo = 0.0

    def procesar_bloque(bloque: list[tuple[str, str]]):
        nonlocal n_frag_calculados, n_tok_calculados, seg_modelo
        frs = fragmentar_bloque([t or "" for _, t in bloque], presupuesto, tok, args.max_fragmentos,
                                args.guardar_texto_fragmentos)
        plano = [(k, f) for k, (_, (_, fl)) in enumerate(zip(bloque, frs)) for f in fl]
        lista_ids = [f[1] for _, f in plano]
        if lista_ids:
            t1 = time.perf_counter()
            probs, errs = predecir(modelo, dispositivo, lista_ids, pre, suf, pad_id, args.lote, n_etq)
            seg_modelo += time.perf_counter() - t1
            n_frag_calculados += len(lista_ids)
            n_tok_calculados += sum(len(x) for x in lista_ids)
        else:
            probs, errs = np.zeros((0, n_etq), np.float32), {}
        malas = {plano[j][0] for j in errs}
        for j, msg in errs.items():
            errores_run[bloque[plano[j][0]][0]] = msg
        filas = []
        for (k, (idx_orig, ids_f, txt)), pr in zip(plano, probs):
            if k in malas:
                continue
            i, _ = bloque[k]
            fila = {"id": i, "hash_texto": hash_de[i], "frag_idx": idx_orig, "n_tokens": len(ids_f) + len(pre) + len(suf),
                    "n_frag_total": frs[k][0]}
            fila.update({f"p_{e}": v for e, v in zip(etiquetas, pr)})
            if args.guardar_texto_fragmentos:
                fila["texto_fragmento"] = txt
            filas.append(fila)
        for k, (i, _) in enumerate(bloque):  # noticias sin ningún fragmento (texto vacío): se registran
            if not frs[k][1] and i not in errores_run:
                fila = {"id": i, "hash_texto": hash_de[i], "frag_idx": 0, "n_tokens": 0, "n_frag_total": 0}
                fila.update({f"p_{e}": np.float32("nan") for e in etiquetas})
                if args.guardar_texto_fragmentos:
                    fila["texto_fragmento"] = ""
                filas.append(fila)
        if filas:
            df = pd.DataFrame(filas)
            for c in ("frag_idx", "n_tokens", "n_frag_total"):
                df[c] = df[c].astype("int32")
            for e in etiquetas:
                df[f"p_{e}"] = df[f"p_{e}"].astype("float32")
            escritura_atomica_parquet(pa.Table.from_pandas(df, preserve_index=False),
                                      cache_dir / f"parte_{time.time_ns()}.parquet")

    interrumpido = False
    if pendientes:
        pend_set = set(pendientes)
        b = barra(total=len(pendientes), desc="emociones", unit="noticia")
        buf: list[tuple[str, str]] = []
        try:
            for ids, textos in iterar_textos(ARCHIVO_PREPARADO):
                for i, t in zip(ids, textos):
                    if i in pend_set:
                        buf.append((i, t))
                while len(buf) >= args.tamano_bloque:
                    bl, buf = buf[:args.tamano_bloque], buf[args.tamano_bloque:]
                    procesar_bloque(bl)
                    if b:
                        b.update(len(bl))
                        b.set_postfix(frag_s=f"{n_frag_calculados / max(seg_modelo, 1e-9):.1f}")
            if buf:
                procesar_bloque(buf)
                if b: b.update(len(buf))
        except KeyboardInterrupt:
            interrumpido = True
        if b: b.close()
        if interrumpido:
            print("\nInterrumpido. El progreso hasta el último bloque quedó guardado: vuelve a ejecutar el mismo comando para continuar.")
            return 130

    # 6) Consolidación: partes -> resultados por fragmento y por noticia
    partes = [pq.read_table(p) for p in sorted(cache_dir.glob("parte_*.parquet"))]
    if not partes:
        print("No hay resultados que consolidar.")
        return 1
    frag = pa.concat_tables(partes, promote_options="default").to_pandas()
    frag = frag[frag["id"].map(hash_de).eq(frag["hash_texto"])]
    frag = frag.drop_duplicates(["id", "hash_texto", "frag_idx"], keep="last")
    frag_ok = frag[frag["n_tokens"] > 0]
    agg = agregar(frag_ok, etiquetas, thr, ekman, args.min_prevalencia)
    tot_frag = frag.groupby("id")["n_frag_total"].max().rename("n_fragmentos").reset_index()

    noticias = meta.merge(tot_frag, on="id", how="left").merge(agg, on="id", how="left")
    estado = np.where(noticias["n_fragmentos_procesados"].notna(), "ok",
                      np.where(noticias["n_fragmentos"].eq(0), "sin_texto", "sin_resultado"))
    noticias.insert(len(meta.columns), "estado", estado)
    noticias["error"] = noticias["id"].map(errores_run)
    noticias = noticias.sort_values(["idioma", "fecha_final", "id"] if "fecha_final" in noticias.columns else ["idioma", "id"])

    columnas_meta_frag = [c for c in ("id", "idioma", "diario", "fecha_final") if c in meta.columns]
    frag_sal = frag.drop(columns=["hash_texto"]).merge(meta[columnas_meta_frag], on="id", how="left")
    frag_sal["detectadas"] = [";".join(np.array(etiquetas)[r]) for r in
                              (frag_sal[[f"p_{e}" for e in etiquetas]].to_numpy() >= thr[None, :])]
    orden_cols = columnas_meta_frag + ["frag_idx", "n_frag_total", "n_tokens", "detectadas"] + \
                 [f"p_{e}" for e in etiquetas] + (["texto_fragmento"] if "texto_fragmento" in frag_sal.columns else [])
    frag_sal = frag_sal[orden_cols].sort_values(["idioma", "id", "frag_idx"] if "idioma" in frag_sal.columns else ["id", "frag_idx"])

    params = {"modelo": args.modelo, "revision": revision, "umbral": modo_umbral, "min_prevalencia": args.min_prevalencia,
              "max_tokens": args.max_tokens, "max_fragmentos": args.max_fragmentos, "algoritmo": ALGORITMO, "huella_cache": huella}
    tabla_n = pa.Table.from_pandas(noticias, preserve_index=False)
    tabla_n = tabla_n.replace_schema_metadata({**(tabla_n.schema.metadata or {}), b"ejecutar_modelo": json.dumps(params).encode()})
    escritura_atomica_parquet(tabla_n, salida_dir / "emociones_por_noticia.parquet")
    escritura_atomica_parquet(pa.Table.from_pandas(frag_sal, preserve_index=False), salida_dir / "emociones_por_fragmento.parquet")
    if args.csv:
        noticias.to_csv(salida_dir / "emociones_por_noticia.csv", index=False, encoding="utf-8-sig")
    if errores_run:
        pd.DataFrame([{"id": k, "error": v} for k, v in errores_run.items()]).to_csv(
            salida_dir / "errores.csv", index=False, encoding="utf-8-sig")

    ok = noticias[noticias["estado"] == "ok"]
    resumen = []
    for idioma, g in ok.groupby("idioma"):
        for e in etiquetas:
            resumen.append({"idioma": idioma, "emocion": e, "pct_noticias_detectada": round(100 * g[f"det_{e}"].mean(), 2),
                            "media_de_prob_media": round(float(g[f"media_{e}"].mean()), 4)})
    pd.DataFrame(resumen).to_csv(salida_dir / "resumen_por_idioma.csv", index=False, encoding="utf-8-sig")

    vel = n_frag_calculados / seg_modelo if seg_modelo > 0 else None
    fr_por_noticia = frag_ok.groupby("id").size().mean() if len(frag_ok) else float("nan")
    meta_json = {
        "generado_utc": datetime.now(timezone.utc).isoformat(), "duracion_s": round(time.time() - t0, 1),
        "parametros": params, "lote": args.lote, "dispositivo": dispositivo, "python": platform.python_version(),
        "torch": torch.__version__, "datos_sha256": sha256_archivo(ARCHIVO_PREPARADO),
        "noticias_objetivo": len(meta), "noticias_ok": int((noticias["estado"] == "ok").sum()),
        "noticias_con_error": int((noticias["estado"] == "sin_resultado").sum()),
        "fragmentos_calculados_en_esta_ejecucion": n_frag_calculados,
        "fragmentos_por_segundo": round(vel, 2) if vel else None, "etiquetas": etiquetas,
    }
    (salida_dir / "ejecucion.json").write_text(json.dumps(meta_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nListo en {time.time() - t0:.0f} s. Noticias ok: {meta_json['noticias_ok']:,} | con error: {meta_json['noticias_con_error']:,}")
    print(f"Fragmentos calculados ahora: {n_frag_calculados:,}" + (f" a {vel:.1f} fragmentos/s" if vel else " (todo se reutilizó)"))
    print(f"Archivos en {salida_dir}:\n  emociones_por_noticia.parquet | emociones_por_fragmento.parquet | resumen_por_idioma.csv | ejecucion.json")
    if len(ok):
        top = pd.Series({e: ok[f"det_{e}"].mean() for e in etiquetas}).sort_values(ascending=False).head(8)
        print("Emociones más detectadas (% de noticias): " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in top.items()))
    if args.muestra and vel and not np.isnan(fr_por_noticia):
        por_idioma = frag_ok.merge(meta[["id", "idioma"]], on="id").groupby(["idioma", "id"]).size().groupby("idioma").mean()
        conteo = pd.read_parquet(ARCHIVO_PREPARADO, columns=["idioma"])["idioma"].value_counts()
        total_est = float(sum(por_idioma.get(i, fr_por_noticia) * n for i, n in conteo.items()))
        print(f"\nESTIMACIÓN ORIENTATIVA para las {total_corpus:,} noticias: ~{total_est:,.0f} fragmentos "
              f"-> ~{total_est / vel / 3600:.1f} horas a {vel:.1f} fragmentos/s (velocidad medida con esta muestra y tu lote={args.lote}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
