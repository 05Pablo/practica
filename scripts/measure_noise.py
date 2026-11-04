"""Mide el ruido de etiquetas del dataset.

Cada párrafo hereda la clase de la categoría de Vikipedio de su artículo, así que
hay párrafos cuya etiqueta no se puede deducir de su texto. El script estima
cuántos hay con tres medidas independientes:

1. Otro idioma: párrafos en los que la mayoría de las palabras no tienen forma
   de Esperanto. Usa el etiquetador de `esperanto_nlp`, que asigna POS por la
   terminación y marca con `X` lo que no reconoce (alemán, ruso, inglés...).
2. Sin señal de su clase: párrafos que no contienen ningún término
   característico de su clase. Un término es característico si aparece en al
   menos MIN_DF párrafos y al menos el MIN_PURITY de ellos son de una clase
   (frecuencias relativas al tamaño de cada clase, para no favorecer a la
   mayoritaria). No depende de ningún modelo.
3. Error persistente: párrafos que un clasificador (TF-IDF + regresión
   logística, como en el cuaderno) falla en TODAS las repeticiones de una
   validación cruzada estratificada. Si cambiar la partición no lo arregla, lo
   más probable es que la etiqueta no encaje con el texto.

Ninguna medida es exacta: son estimaciones del ruido. La intersección de las
medidas da los casos más claros.

No sabe nada de la temática: funciona con cualquier CSV que cumpla el contrato
de datos (`text;class`).

Uso:
    pip install -e ".[analysis]"
    python scripts/measure_noise.py
    python scripts/measure_noise.py --out noise.csv   # detalle por párrafo
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import esperanto_nlp

ROOT = Path(__file__).resolve().parent.parent

FOREIGN_X_RATIO = 0.3   # proporción mínima de palabras con POS "X" (sin nombres propios)
MIN_FOREIGN_WORDS = 4   # palabras mínimas para juzgar el idioma
MIN_DF = 5              # párrafos mínimos en los que aparece un término característico
MIN_PURITY = 0.8        # proporción mínima de una clase para un término característico

nlp = esperanto_nlp.load()


def identity(tokens):
    return tokens


def analyze(texts):
    """Lemas útiles (como en el cuaderno) y proporción de palabras con POS X.

    Los nombres propios no cuentan para el idioma: las referencias
    bibliográficas en otros idiomas están llenas de mayúsculas y el etiquetador
    las marca como PROPN, lo que escondería que el texto no es Esperanto.
    """
    lemmas, x_ratio = [], []
    for text in texts:
        doc = nlp(text)
        lemmas.append([t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop])
        words = [t for t in doc if t.is_alpha and t.pos_ != "PROPN"]
        n_x = sum(1 for t in words if t.pos_ == "X")
        x_ratio.append(n_x / len(words) if len(words) >= MIN_FOREIGN_WORDS else 0.0)
    return lemmas, x_ratio


def characteristic_terms(lemmas, labels, classes):
    """Términos que aparecen casi solo en una clase: {clase: set(términos)}."""
    size = Counter(labels)
    df = defaultdict(Counter)            # término -> {clase: nº de párrafos}
    for terms, label in zip(lemmas, labels):
        for term in set(terms):
            df[term][label] += 1

    result = {c: set() for c in classes}
    for term, by_class in df.items():
        if sum(by_class.values()) < MIN_DF:
            continue
        rel = {c: by_class[c] / size[c] for c in classes}   # frecuencia relativa
        best = max(rel, key=rel.get)
        if rel[best] / sum(rel.values()) >= MIN_PURITY:
            result[best].add(term)
    return result


def persistent_errors(lemmas, labels, seed, folds, repeats):
    """Tasa de error de cada párrafo en una validación cruzada repetida."""
    y = np.array(labels)
    errors = np.zeros(len(labels))
    wrong_as = [Counter() for _ in labels]
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for train, test in cv.split(lemmas, y):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer=identity)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=seed)),
        ])
        model.fit([lemmas[i] for i in train], y[train])
        pred = model.predict([lemmas[i] for i in test])
        for i, p in zip(test, pred):
            if p != y[i]:
                errors[i] += 1
                wrong_as[i][p] += 1
    return errors / repeats, wrong_as


def pct(n, total):
    return f"{n:5d}  ({100 * n / total:5.1f} %)" if total else f"{n:5d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=ROOT / "data" / "dataset.csv", type=Path)
    parser.add_argument("--out", type=Path, help="CSV con el detalle por párrafo (opcional)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--examples", default=3, type=int, help="ejemplos por medida")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.dataset, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    texts = [r["text"] for r in rows]
    labels = [r["class"] for r in rows]
    classes = sorted(set(labels))
    n = len(texts)

    print(f"Dataset: {n} párrafos, clases: {', '.join(classes)}\n")
    lemmas, x_ratio = analyze(texts)

    foreign = [r >= FOREIGN_X_RATIO for r in x_ratio]

    terms = characteristic_terms(lemmas, labels, classes)
    no_signal = [not (set(l) & terms[c]) for l, c in zip(lemmas, labels)]

    print(f"Validación cruzada: {args.folds} folds x {args.repeats} repeticiones...\n")
    error_rate, wrong_as = persistent_errors(lemmas, labels, args.seed, args.folds, args.repeats)
    always_wrong = [e == 1.0 for e in error_rate]

    noisy = [a and (b or c) for a, b, c in zip(always_wrong, foreign, no_signal)]
    any_flag = [a or b or c for a, b, c in zip(always_wrong, foreign, no_signal)]

    measures = [
        (f"Otro idioma (POS X ≥ {FOREIGN_X_RATIO:.0%})", foreign),
        ("Sin términos de su clase", no_signal),
        ("Error en todas las repeticiones", always_wrong),
        ("Error persistente + otra señal", noisy),
        ("Alguna de las tres medidas", any_flag),
    ]

    # Resumen global y por clase
    header = f"{'Medida':34s}{'Total':>16s}" + "".join(f"{c:>14s}" for c in classes)
    print(header)
    print("-" * len(header))
    size = Counter(labels)
    for name, flags in measures:
        line = f"{name:34s}{pct(sum(flags), n):>16s}"
        for c in classes:
            k = sum(1 for flag, l in zip(flags, labels) if flag and l == c)
            line += f"{100 * k / size[c]:13.1f}%"
        print(line)

    print(f"\nTérminos característicos por clase: "
          + ", ".join(f"{c}={len(terms[c])}" for c in classes))
    print(f"Error medio por párrafo en la validación cruzada: {error_rate.mean():.3f}")

    # A qué clase se van los errores persistentes
    flows = Counter((labels[i], wrong_as[i].most_common(1)[0][0])
                    for i in range(n) if always_wrong[i])
    if flows:
        print("\nErrores persistentes (clase del dataset -> clase predicha):")
        for (real, pred), k in flows.most_common():
            print(f"  {real:>14s} -> {pred:<14s} {k}")

    rng = random.Random(args.seed)
    for name, flags in measures[:3]:
        idx = [i for i, flag in enumerate(flags) if flag]
        if not idx:
            continue
        print(f"\nEjemplos — {name}:")
        for i in rng.sample(idx, min(args.examples, len(idx))):
            print(f"  [{labels[i]}] {texts[i][:160]}{'…' if len(texts[i]) > 160 else ''}")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["row", "class", "x_ratio", "foreign", "no_signal",
                             "cv_error_rate", "always_wrong", "predicted_as"])
            for i in range(n):
                writer.writerow([
                    i, labels[i], round(x_ratio[i], 3), int(foreign[i]), int(no_signal[i]),
                    round(float(error_rate[i]), 2), int(always_wrong[i]),
                    wrong_as[i].most_common(1)[0][0] if wrong_as[i] else "",
                ])
        print(f"\nDetalle por párrafo guardado en {args.out}")


if __name__ == "__main__":
    main()
