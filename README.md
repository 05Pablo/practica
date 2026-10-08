# practica
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
