import glob
import re
import pandas as pd

# 1. Patrones de ruido detectados en la extracción con trafilatura
RUIDO_REGEX = re.compile(
    r"^\s*(passer la publicit[ée]|publicit[ée],?\s*votre contenu continue.*|"
    r"leia tamb[ée]m|lire aussi|leer tambi[ée]n|read more|weiterlesen|"
    r"iniciar sesi[óo]n|log in|sign in|connexion|anmelden|"
    r"suscr[ií]bete|subscribe|abonnez-vous|abonnieren|assine|"
    r"compartir|share|partager|teilen)\s*[:\-–]?\s*$",
    re.IGNORECASE,
)

# 2. Términos clave de IA en los 5 idiomas (es, en, pt, fr, de)
TERMINOS_IA = re.compile(
    r"\b(ia|ai|ki|inteligencia artificial|artificial intelligence|"
    r"intelig[êe]ncia artificial|intelligence artificielle|k[üu]nstliche intelligenz|"
    r"chatgpt|openai|gpt-?\d*|llm|llms|large language model\w*|"
    r"machine learning|aprendizaje autom[áa]tico|aprendizado de m[áa]quina|"
    r"apprentissage automatique|maschinelles lernen|deep learning|"
    r"generative ai|ia generativa|gemini|claude|copilot|midjourney|deepmind|anthropic)\b",
    re.IGNORECASE,
)

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    lineas = [
        l.strip() for l in texto.splitlines()
        if l.strip() and not RUIDO_REGEX.match(l.strip())
    ]
    return "\n".join(lineas)

def fragmentar_contexto_ia(texto: str, max_palabras: int = 330) -> list[str]:
    texto_limpio = limpiar_texto(texto)
    oraciones = [
        o.strip() for o in re.split(r"(?<=[.!?])\s+|\n+", texto_limpio)
        if len(o.strip()) > 15
    ]
    if not oraciones:
        return []

    # Conservar la primera oración (titular incrustado) + ventana [-1, +1] alrededor de menciones de IA
    indices_relevantes = {0}
    for i, oracion in enumerate(oraciones):
        if TERMINOS_IA.search(oracion):
            for j in range(max(0, i - 1), min(len(oraciones), i + 2)):
                indices_relevantes.add(j)

    seleccionadas = [oraciones[i] for i in sorted(indices_relevantes)] or oraciones

    # Agrupar en chunks <= max_palabras para respetar el límite de 512 tokens
    chunks, actual, conteo = [], [], 0
    for oracion in seleccionadas:
        palabras = oracion.split()
        if conteo + len(palabras) > max_palabras and actual:
            chunks.append(" ".join(actual))
            actual, conteo = palabras, len(palabras)
        else:
            actual.extend(palabras)
            conteo += len(palabras)
    if actual:
        chunks.append(" ".join(actual))
    return chunks

# Cargar los 5 archivos noticias_<idioma>.parquet y generar tabla de fragmentos
archivos = sorted(glob.glob("entrega/noticias_*.parquet"))
registros = []

for ruta in archivos:
    df = pd.read_parquet(ruta)
    for row in df.itertuples(index=False):
        chunks = fragmentar_contexto_ia(row.texto)
        for idx_chunk, texto_chunk in enumerate(chunks):
            registros.append({
                "id": row.id,
                "chunk_id": idx_chunk,
                "n_chunks": len(chunks),
                "idioma": row.idioma,
                "diario": row.diario,
                "tipo": row.tipo,
                "anio_pub": row.anio_pub,
                "mes_pub": row.mes_pub,
                "origen_url": row.origen_url,
                "texto_modelo": texto_chunk,
            })

df_chunks = pd.DataFrame(registros)
df_chunks.to_parquet("entrega/corpus_listo_inferencia.parquet", index=False)
print(f"Noticias procesadas: {df_chunks['id'].nunique():,} | Fragmentos totales: {len(df_chunks):,}")