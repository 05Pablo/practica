# practica

[![Tests](https://github.com/05Pablo/practica/actions/workflows/tests.yml/badge.svg)](https://github.com/05Pablo/practica/actions/workflows/tests.yml)

Practica final de la asignatura 'Minería de Textos'

Clasificación de textos en Esperanto por temática: construcción del dataset,
librería de NLP propia basada en reglas y cuaderno de clasificación.

## Temática

Religiones del mundo:

| Clase | Categoría de Wikipedia (eo) |
|---|---|
| `kristanismo` | [Kategorio:Kristanismo](https://eo.wikipedia.org/wiki/Kategorio:Kristanismo) |
| `islamo` | [Kategorio:Islamo](https://eo.wikipedia.org/wiki/Kategorio:Islamo) |
| `judismo` | [Kategorio:Judismo](https://eo.wikipedia.org/wiki/Kategorio:Judismo) |

## Estructura

```
src/esperanto_nlp/   Librería de NLP (Fase 2)
tests/               Pruebas unitarias (pytest)
scripts/             Script de construcción del dataset (Fase 1)
data/                Dataset (CSV UTF-8, separador ;, columnas text;class)
notebooks/           Cuaderno de entrenamiento y clasificación (Fase 3)
```

## Instalación

```bash
pip install git+https://github.com/05Pablo/practica.git
```

## Uso de la librería

```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La judoj preĝas en la sinagogo. Ĉu vi?")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.morph, token.is_stop)
# La        la        DET
# judoj     judo      NOUN   Case=Nom|Number=Plur
# preĝas    preĝi     VERB   Mood=Ind|Tense=Pres|VerbForm=Fin
# ...

for sent in doc.sents:
    print(sent.text)
```

El pipeline (`nlp.pipe_names`) tiene tres componentes basados en reglas:

| Componente | Asigna | Reglas |
|---|---|---|
| `tagger` | `pos_`, `morph` | Terminaciones (-o, -a, -e, -i, -as...), léxico cerrado de palabras funcionales y correlativos, contexto (DET/PRON, nombres propios) |
| `lemmatizer` | `lemma_` | Elimina la flexión: plural, acusativo y tiempo verbal (`estis` → `esti`). No elimina la derivación (`konataj` → `konata`) |
| `stopwords` | `is_stop` | Lista de stop-words aplicada a la forma y al lema |

Las categorías siguen las etiquetas [Universal Dependencies](https://universaldependencies.org/u/pos/).

La lista de stop-words parte de `vortoj.json` del repositorio
[nlp-esperantilo](https://github.com/jparisu/nlp-esperantilo) (Apache-2.0) y se
amplía por reglas con la tabla de correlativos y las formas de los pronombres.

Para desarrollo (instalación editable con dependencias de pruebas):

```bash
pip install -e ".[dev]"
pytest
```
## Cuaderno de clasificación (Fase 3)

El cuaderno [`notebooks/clasificador.ipynb`](notebooks/clasificador.ipynb) entrena y
evalúa un clasificador de la temática de párrafos en Esperanto usando la librería
`esperanto_nlp`. Se ejecuta en Google Colab sin ficheros locales: instala la librería
desde GitHub y descarga el dataset por URL.

Secciones:

1. **Instalación y carga**: `pip install` desde GitHub y lectura del CSV por URL.
2. **Análisis del dataset**: párrafos por clase, longitudes y desequilibrio.
3. **Preprocesado**: lemas en minúsculas de las palabras alfabéticas que no son
   stop-words, con nuestra librería.
4. **TF-IDF**: `TfidfVectorizer(analyzer=preprocess)`.
5. **Entrenamiento**: split estratificado 80/20 con `random_state=42` y comparación de
   Naive Bayes, Complement Naive Bayes y regresión logística (`class_weight="balanced"`).
6. **Evaluación**: accuracy, F1 macro, matriz de confusión, informe por clase y análisis
   de errores.
7. **Clasificación web**: descarga de páginas de Vikipedio con `requests` y
   `BeautifulSoup` y probabilidad de cada clase con `predict_proba`. Si una descarga
   falla (sin conexión, *timeout*, error HTTP), el cuaderno muestra un aviso y continúa.

### Usarlo con otro dataset

Las clases se obtienen de `df["class"].unique()`, así que no hay nada escrito a mano
sobre esta temática. Para usar el cuaderno con otro dataset basta con cambiar
`DATASET_URL` (CSV con separador `;` y columnas `text;class`) y la lista `PAGINAS` de la
sección 7.

### Resultados

Conjunto de test: 20 % del dataset, estratificado, `random_state=42`.

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Naive Bayes | 0.798 | 0.781 |
| Complement Naive Bayes | 0.887 | 0.884 |
| Regresión logística (`class_weight="balanced"`) | 0.895 | 0.892 |

- **Naive Bayes** tiende a asignar la clase mayoritaria (`kristanismo`) cuando hay dudas.
  Complement NB y la regresión logística con pesos corrigen ese sesgo, por lo que el
  problema era el desequilibrio de clases más que el algoritmo.
- La confusión principal es entre `judismo` y `kristanismo`, que comparten vocabulario y
  personajes bíblicos.

### Limitaciones

- **Ruido de etiquetas.** Cada párrafo hereda la clase de la categoría de Vikipedio del
  artículo, así que hay párrafos que no distinguen la clase (biografías de personas,
  citas en otros idiomas) y el modelo los clasifica mal.
- **Clases cerradas.** El modelo solo conoce las tres religiones. Una página de otra
  temática se asignará igualmente a la clase más probable.
- **Fuente.** El dataset procede de Vikipedio; en otros estilos de texto (blogs, noticias)
  el rendimiento puede ser menor.
- **Lematizador.** Elimina la flexión, no la derivación, por lo que formas derivadas del
  mismo radical cuentan como términos distintos.
