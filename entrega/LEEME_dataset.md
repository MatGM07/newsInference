# Dataset: noticias sobre inteligencia artificial, 2021-2026

Corpus de **78,371 noticias** sobre IA publicadas entre 2021 y octubre de 2026 en 37 diarios de 5 idiomas
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
| `anio_gdelt` | año con el que se muestreó la noticia (puede diferir de `anio_pub` en 0.0% de los casos) |
| `titulo` | título de la noticia. **Disponible en 0% de las filas**; si está vacío, no se guardó al descargar |
| `texto` | cuerpo de la noticia extraído de la página |
| `n_palabras` | palabras del texto (mínimo 150) |
| `n_menciones_ia` | veces que el texto menciona términos de IA (ChatGPT, OpenAI, inteligencia artificial, KI, IA...) |
| `origen_url` | `match` (la URL traía una palabra clave de IA), `tech_sin_match` o `control` (la URL no la traía) |
| `via` | `directo` o `wayback` (descargada de Wayback Machine) |
| `url` | enlace original |

## Cobertura
| idioma | noticias | diarios | palabras_mediana |
|---|---|---|---|
| de | 9,593 | 9 | 429 |
| en | 20,136 | 6 | 605 |
| es | 13,053 | 8 | 651 |
| fr | 12,069 | 7 | 482 |
| pt | 23,520 | 7 | 536 |

Noticias por idioma y año:

| idioma | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| de | 149 | 244 | 2,914 | 2,740 | 1,579 | 1,967 |
| en | 425 | 910 | 4,463 | 4,731 | 4,606 | 5,001 |
| es | 176 | 26 | 2,519 | 3,242 | 3,619 | 3,471 |
| fr | 198 | 244 | 2,687 | 2,502 | 3,315 | 3,123 |
| pt | 557 | 170 | 4,633 | 6,172 | 6,200 | 5,788 |

% de noticias de diarios de tecnología por idioma y año:

| idioma | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| de | 81.0 | 74.0 | 67.0 | 64.0 | 63.0 | 56.0 |
| en | 90.0 | 92.0 | 81.0 | 83.0 | 81.0 | 75.0 |
| es | 35.0 | 0.0 | 20.0 | 24.0 | 24.0 | 23.0 |
| fr | 60.0 | 74.0 | 69.0 | 58.0 | 52.0 | 49.0 |
| pt | 53.0 | 35.0 | 22.0 | 32.0 | 29.0 | 25.0 |

## Limitaciones que hay que tener presentes
1. **La cobertura de IA crece muchísimo desde 2023** (efecto ChatGPT) y, además, GDELT indexa menos diarios en 2021-2022. Por eso 2021-2022
   tienen muchas menos noticias y varios idiomas-año quedaron bajo su meta. Analiza **proporciones** (por ejemplo % negativo por trimestre),
   no conteos, y agrupa 2021-2022 en periodos más largos.
2. **La composición de fuentes cambia con el tiempo** (ver `composicion_diarios.csv` y el % de tecnología de arriba). Un cambio de tono puede venir
   de un cambio de fuente y no del discurso. Incluye `diario` o `tipo` como variable de control o compara dentro del mismo tipo de diario.
3. **Tope por diario:** se limitó cada diario al 30% de la *meta* de su idioma-año, no de lo que finalmente se entregó. Como varios idiomas-año
   quedaron por debajo de su meta, un diario puede pesar más del 30% del total entregado (hasta 80% en algún idioma-año; ver
   `composicion_diarios.csv`). Hay que corregirlo en el análisis (por diario o por tipo de diario), no darlo por balanceado.
4. **Selección:** la mayoría de noticias se encontró porque su URL traía una palabra clave de IA; se incluyó una muestra pequeña sin ella (`control`)
   y, en diarios de tecnología, URLs sin palabra clave como relleno. Se aceptó una noticia si el texto menciona términos de IA 3 o más veces
   (o, cuando el título se guardó, si el título la mencionaba), así que puede haber algunos falsos positivos (noticias donde la IA es secundaria).
5. **Calidad del texto:** extraído automáticamente con trafilatura; puede traer restos de menús, avisos de cookies o textos truncados por muros de pago.
   Revisa `muestra_revision.csv` antes de lanzar el modelo sobre todo.
6. **Muestra, no censo:** representa lo que GDELT indexó de 37 diarios, no toda la prensa.
7. **Fechas:** `fecha_final` viene de GDELT (fecha en que GDELT registró la noticia, normalmente muy cerca de la publicación), no de la página. Es adecuada para agrupar por mes o trimestre, pero no para análisis al día.
8. **Títulos:** solo 0% de las filas tienen `titulo`. No se podrá analizar el tono de los titulares salvo que se recuperen aparte (volviendo a visitar las URLs de una muestra).

## Siguientes pasos sugeridos (sentimiento y emociones)
1. **Revisar la muestra** (`muestra_revision.csv`) y decidir si hay que filtrar plantillas o textos cortos.
2. **Elegir el modelo.** Mi recuerdo, que hay que verificar en la documentación: `pysentimiento` cubre sentimiento en es/en/pt/it y emociones en
   es/en/it, es decir, **no cubre francés ni alemán**. Para esos hace falta un modelo multilingüe (por ejemplo uno tipo XLM-RoBERTa para sentimiento).
   Conviene usar **el mismo modelo en los 5 idiomas** para que los resultados sean comparables, o al menos validar con una muestra etiquetada a mano.
3. **Textos largos.** Los modelos suelen aceptar 128-512 tokens y las noticias son más largas (mediana 560 palabras). Opciones: título + primeros
   párrafos, o dividir en fragmentos y agregar los resultados (promedio o proporción) por noticia.
4. **Ojo con el dominio:** estos modelos suelen estar entrenados con tuits; las noticias son más neutras, así que la mayoría saldrá neutral. Mejor
   analizar la distribución y los cambios relativos que la etiqueta de cada noticia.
5. **Temas:** extracción de temas por idioma (por ejemplo BERTopic con un modelo multilingüe) y luego cruzar tema × tiempo × sentimiento.
6. **Resultado esperado:** una tabla con `id` y las columnas de sentimiento/emociones/tema, para unir con este dataset por `id`.
