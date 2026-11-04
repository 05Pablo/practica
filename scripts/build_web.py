"""Genera la web del proyecto (GitHub Pages) en una carpeta estática.

Pasos:
1. Copia los ficheros estáticos de `web/`.
2. Empaqueta la librería `esperanto_nlp` en un zip que el navegador carga con
   Pyodide (Python en WebAssembly), para usarla sin servidor.
3. Entrena los mismos modelos que el cuaderno (mismo preprocesado, partición
   80/20 estratificada y semilla) y exporta:
   - `data/model.json`: vocabulario, IDF y coeficientes de la regresión
     logística, para clasificar en el navegador.
   - `data/results.json`: estadísticas del dataset y métricas de evaluación.

El script no sabe nada de la temática: las clases salen del CSV, así que sirve
con cualquier dataset que cumpla el contrato de datos (`text;class`).

Uso:
    pip install -e . -r web/requirements.txt
    python scripts/build_web.py            # genera _site/
    python -m http.server -d _site 8000    # vista previa en localhost:8000
"""

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

import esperanto_nlp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web" / "py"))
import web_demo  # noqa: E402  (mismo clasificador que usa el navegador)

LIBRARY_DIR = ROOT / "src" / "esperanto_nlp"

nlp = esperanto_nlp.load()


def preprocess(text):
    """Igual que en el cuaderno: lemas en minúsculas, sin stop-words."""
    return [t.lemma_.lower() for t in nlp(text) if t.is_alpha and not t.is_stop]


def identity(tokens):
    return tokens


def read_dataset(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    texts = [row["text"] for row in rows]
    labels = [row["class"] for row in rows]
    return texts, labels


def make_models(seed):
    # Los textos llegan ya preprocesados (lista de lemas), así que el analyzer
    # es la identidad: el resultado es el mismo que analyzer=preprocess.
    def tfidf():
        return TfidfVectorizer(analyzer=identity)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Complement NB": Pipeline([("tfidf", tfidf()), ("clf", ComplementNB())]),
        "Regresión logística": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=seed)),
        ]),
    }


def export_model(pipeline):
    """Vocabulario, IDF y coeficientes de un Pipeline TF-IDF + regresión logística."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    coef, intercept = clf.coef_, clf.intercept_
    if coef.shape[0] == 1:
        # Caso binario: sklearn guarda un único vector. Con [0, w] el softmax
        # de dos clases equivale a la sigmoide que usa sklearn.
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    return {
        "classes": [str(c) for c in clf.classes_],
        "terms": [str(t) for t in tfidf.get_feature_names_out()],
        "idf": [round(float(v), 6) for v in tfidf.idf_],
        "coef": [[round(float(v), 6) for v in row] for row in coef],
        "intercept": [round(float(v), 6) for v in intercept],
    }


def length_histogram(texts, labels, classes, bins):
    hist = {c: [0] * (len(bins) + 1) for c in classes}
    for text, label in zip(texts, labels):
        n = len(text.split())
        k = next((i for i, edge in enumerate(bins) if n < edge), len(bins))
        hist[label][k] += 1
    return hist


def build_results(texts, labels, tokens, seed):
    classes = sorted(set(labels))
    idx = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed)
    tok_train = [tokens[i] for i in train_idx]
    tok_test = [tokens[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    models = make_models(seed)
    comparison, confusions, predictions = [], {}, {}
    for name, model in models.items():
        model.fit(tok_train, y_train)
        pred = model.predict(tok_test)
        predictions[name] = pred
        comparison.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        })
        confusions[name] = confusion_matrix(y_test, pred, labels=classes).tolist()

    best = max(comparison, key=lambda r: r["f1_macro"])["model"]
    best_model = models[best]
    report = classification_report(y_test, predictions[best], labels=classes,
                                   output_dict=True, zero_division=0)

    # Términos con más peso por clase (solo si el mejor modelo es lineal).
    top_terms = {}
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "coef_") and clf.coef_.shape[0] == len(classes):
        terms = best_model.named_steps["tfidf"].get_feature_names_out()
        for k, c in enumerate(clf.classes_):
            order = np.argsort(clf.coef_[k])[::-1][:15]
            top_terms[str(c)] = [[str(terms[i]), round(float(clf.coef_[k][i]), 3)]
                                 for i in order]

    # Ejemplos de test correctos e incorrectos con sus probabilidades.
    proba = best_model.predict_proba(tok_test)
    rng = random.Random(seed)
    right = [i for i, p in enumerate(predictions[best]) if p == y_test[i]]
    wrong = [i for i, p in enumerate(predictions[best]) if p != y_test[i]]

    def example(i):
        return {
            "text": texts[test_idx[i]][:400],
            "real": y_test[i],
            "pred": str(predictions[best][i]),
            "proba": [round(float(p), 4) for p in proba[i]],
        }

    examples = {
        "correct": [example(i) for i in rng.sample(right, min(4, len(right)))],
        "wrong": [example(i) for i in rng.sample(wrong, min(4, len(wrong)))],
    }
    error_pairs = Counter((y_test[i], str(predictions[best][i])) for i in wrong)

    # Muestras del dataset por clase para la sección de exploración.
    samples = {}
    for c in classes:
        pool = [t for t, l in zip(texts, labels) if l == c and 25 <= len(t.split()) <= 90]
        samples[c] = rng.sample(pool, min(6, len(pool)))

    bins = [20, 40, 60, 80, 100, 150, 200]
    counts = Counter(labels)
    return best_model, {
        "classes": classes,
        "n_paragraphs": len(texts),
        "class_counts": {c: counts[c] for c in classes},
        "median_words": {c: int(np.median([len(t.split()) for t, l in zip(texts, labels) if l == c]))
                         for c in classes},
        "length_bins": bins,
        "length_hist": length_histogram(texts, labels, classes, bins),
        "samples": samples,
        "split": {"train": len(train_idx), "test": len(test_idx), "seed": seed},
        "comparison": comparison,
        "best_model": best,
        "confusion": confusions,
        "report": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                   for c in classes},
        "top_terms": top_terms,
        "examples": examples,
        "error_pairs": [[r, p, n] for (r, p), n in error_pairs.most_common()],
    }


def count_articles(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return dict(Counter(row["class"] for row in csv.DictReader(f, delimiter=";")))


def zip_library(dest):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(LIBRARY_DIR.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, Path("esperanto_nlp") / path.relative_to(LIBRARY_DIR))


def check_browser_classifier(model_json, pipeline, sample):
    """Comprueba que el clasificador en Python puro da lo mismo que sklearn."""
    clf = web_demo.Classifier(model_json)
    expected = pipeline.predict_proba(sample)
    got = np.array([clf.predict_tokens(tokens)["proba"] for tokens in sample])
    diff = float(np.abs(expected - got).max())
    if diff > 1e-3:
        raise SystemExit(f"El clasificador del navegador difiere de sklearn ({diff:.4f})")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=ROOT / "data" / "dataset.csv", type=Path)
    parser.add_argument("--sources", default=ROOT / "data" / "sources.csv", type=Path)
    parser.add_argument("--out", default=ROOT / "_site", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(ROOT / "web", args.out, ignore=shutil.ignore_patterns(
        "__pycache__", "requirements.txt"))
    (args.out / "data").mkdir(exist_ok=True)
    zip_library(args.out / "esperanto_nlp.zip")

    print("Preprocesando el dataset con esperanto_nlp...")
    texts, labels = read_dataset(args.dataset)
    tokens = [preprocess(t) for t in texts]

    print("Entrenando y evaluando los modelos...")
    model, results = build_results(texts, labels, tokens, args.seed)
    results["articles"] = count_articles(args.sources)
    results["library_version"] = esperanto_nlp.__version__

    model_json = export_model(model)
    diff = check_browser_classifier(model_json, model, tokens[:300])
    print(f"Clasificador del navegador verificado (diferencia máx. {diff:.2e})")

    for name, data in (("model.json", model_json), ("results.json", results)):
        with open(args.out / "data" / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    for row in results["comparison"]:
        print(f"  {row['model']:22s} accuracy={row['accuracy']:.3f}  F1 macro={row['f1_macro']:.3f}")
    print(f"Web generada en {args.out}")


if __name__ == "__main__":
    main()
