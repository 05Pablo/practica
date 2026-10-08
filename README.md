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

Para desarrollo (instalación editable con dependencias de pruebas):

```bash
pip install -e ".[dev]"
pytest
```
