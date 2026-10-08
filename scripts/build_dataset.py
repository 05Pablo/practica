"""Construye el dataset de párrafos en Esperanto a partir de Wikipedia (eo).

Para cada clase se recorren los artículos de una categoría de Wikipedia (y sus
subcategorías hasta una profundidad dada), se descarga el texto plano de cada
artículo, se divide en párrafos y cada párrafo hereda la clase del artículo.

Salidas:
    data/dataset.csv   Contrato de datos: UTF-8, separador ';', columnas text;class
    data/sources.csv   Trazabilidad: artículo, clase, revisión y nº de párrafos

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-articles 100 --depth 0
"""

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# Clase del dataset -> categoría de Wikipedia en Esperanto.
# Es el único sitio que hay que tocar para cambiar de temática.
CATEGORIES = {
    "kristanismo": "Kristanismo",
    "islamo": "Islamo",
    "judismo": "Judismo",
}

API_URL = "https://eo.wikipedia.org/w/api.php"
USER_AGENT = "practica-mineria-textos/0.1 (https://github.com/05Pablo/practica)"

# Secciones que no aportan texto temático (bibliografía, enlaces...).
EXCLUDED_SECTIONS = {
    "vidu ankaŭ", "referencoj", "notoj", "piednotoj", "notoj kaj referencoj",
    "eksteraj ligiloj", "eksteraj ligoj", "ligiloj", "bibliografio",
    "literaturo", "fontoj",
}

# Párrafos más cortos suelen ser pies de foto, elementos de lista o títulos.
MIN_WORDS = 15

HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*=+$")


class WikiClient:
    """Cliente mínimo del API de MediaWiki con pausas y reintentos."""

    def __init__(self, delay=0.5, retries=5):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.retries = retries

    def get(self, **params):
        params.update(format="json", formatversion=2)
        for attempt in range(self.retries):
            time.sleep(self.delay)
            try:
                response = self.session.get(API_URL, params=params, timeout=30)
                if response.status_code == 429:
                    # Demasiadas peticiones: esperar cada vez más (backoff).
                    wait = 10 * (attempt + 1)
                    print(f"  429 recibido, esperando {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                print(f"  Error de red ({error}), reintentando...", file=sys.stderr)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"El API no respondió tras {self.retries} intentos: {params}")


def get_category_articles(client, category, depth):
    """Devuelve los títulos de artículos de una categoría y sus subcategorías."""
    articles = set()
    visited = set()
    frontier = [f"Kategorio:{category}"]

    for _ in range(depth + 1):
        next_frontier = []
        for cat in frontier:
            if cat in visited:
                continue
            visited.add(cat)
            params = {
                "action": "query", "list": "categorymembers", "cmtitle": cat,
                "cmtype": "page|subcat", "cmlimit": 500,
            }
            while True:
                data = client.get(**params)
                for member in data["query"]["categorymembers"]:
                    if member["ns"] == 0:
                        articles.add(member["title"])
                    elif member["ns"] == 14:
                        next_frontier.append(member["title"])
                if "continue" not in data:
                    break
                params.update(data["continue"])
        frontier = next_frontier

    return articles


def get_article(client, title):
    """Devuelve (texto plano, id de revisión) de un artículo, o None si falla."""
    data = client.get(
        action="query", prop="extracts|info", explaintext=1, titles=title,
    )
    page = data["query"]["pages"][0]
    text = page.get("extract")
    if not text:
        return None
    return text, page.get("lastrevid")


def split_paragraphs(text, min_words=MIN_WORDS):
    """Divide el texto de un artículo en párrafos limpios.

    Ignora los encabezados, las secciones de EXCLUDED_SECTIONS (con todas sus
    subsecciones) y los párrafos de menos de `min_words` palabras.
    """
    paragraphs = []
    skip_level = None  # nivel del encabezado excluido en el que estamos

    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and heading.group(2).lower() in EXCLUDED_SECTIONS:
                skip_level = level
            continue

        if skip_level is None and len(line.split()) >= min_words:
            paragraphs.append(line)

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=1,
                        help="profundidad de subcategorías a recorrer (por defecto 1)")
    parser.add_argument("--max-articles", type=int, default=200,
                        help="máximo de artículos por clase (por defecto 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    client = WikiClient()
    rng = random.Random(args.seed)

    # 1. Títulos de artículos por clase.
    titles_by_class = {}
    for label, category in CATEGORIES.items():
        print(f"Recorriendo Kategorio:{category}...")
        titles_by_class[label] = get_category_articles(client, category, args.depth)
        print(f"  {len(titles_by_class[label])} artículos")

    # Un artículo en varias categorías tendría una etiqueta ambigua: se descarta.
    counts = Counter(t for titles in titles_by_class.values() for t in titles)
    ambiguous = {t for t, n in counts.items() if n > 1}
    print(f"Descartados {len(ambiguous)} artículos presentes en varias clases")

    # 2. Descarga y división en párrafos (muestra aleatoria reproducible).
    rows = []
    sources = []
    for label, titles in titles_by_class.items():
        candidates = sorted(titles - ambiguous)
        rng.shuffle(candidates)
        selected = candidates[:args.max_articles]
        print(f"Descargando {len(selected)} artículos de '{label}'...")

        for i, title in enumerate(selected, 1):
            try:
                article = get_article(client, title)
            except RuntimeError as error:
                print(f"  Omitido '{title}': {error}", file=sys.stderr)
                continue
            if article is None:
                continue
            text, revid = article
            paragraphs = split_paragraphs(text)
            rows.extend((p, label) for p in paragraphs)
            sources.append((title, label, revid, len(paragraphs)))
            if i % 25 == 0:
                print(f"  {i}/{len(selected)}")

    # 3. Párrafos repetidos (plantillas, textos copiados entre artículos).
    #    Si se repiten con clases distintas, la etiqueta es ambigua: fuera todos.
    labels_by_text = {}
    for text, label in rows:
        labels_by_text.setdefault(text, set()).add(label)
    rows = [
        (text, labels.pop())
        for text, labels in labels_by_text.items()
        if len(labels) == 1
    ]

    # 4. Escritura según el contrato de datos.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "dataset.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["text", "class"])
        writer.writerows(rows)

    with open(args.output_dir / "sources.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["title", "class", "revid", "paragraphs"])
        writer.writerows(sorted(sources))

    print("\nPárrafos por clase:")
    for label, n in sorted(Counter(label for _, label in rows).items()):
        print(f"  {label:15} {n}")
    print(f"Total: {len(rows)} -> {args.output_dir / 'dataset.csv'}")


if __name__ == "__main__":
    main()
