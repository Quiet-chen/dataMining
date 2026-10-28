#!/usr/bin/env python3
"""
Evaluate MultiVerS citation-accuracy predictions.

The repo does not ship an evaluation script, so this implements the metrics
described in the paper (Section 2.6): per-class precision / recall / F1,
micro-F1 and macro-F1 over ACCURATE / NOT_ACCURATE / IRRELEVANT.

Two evaluation levels are reported:

pair level
    Every (citation, paragraph) pair is scored directly.  This is the level the
    released paragraph-candidate data actually trains on.

citation level
    For each citation, pick the candidate paragraph with the largest
    P(ACCURATE) + P(NOT_ACCURATE) mass (i.e. the paragraph most likely to be
    the evidence paragraph) and use its predicted label.  If that paragraph is
    predicted IRRELEVANT, the citation is predicted IRRELEVANT.  Gold labels are
    read from Data/annotations.zip, which retains the citation-level label even
    for ETIQUETTE / IRRELEVANT citations that have no evidence paragraph.

Example:
    python eval_predictions.py \
        --predictions predictions/test_predictions.jsonl \
        --claims data_train/target/scifact_10/claims_test.jsonl \
        --annotations ../Data/annotations.zip \
        --output predictions/eval_results.json
"""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

LABELS = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]

ACCURATE_FINE = {"ACCURATE", "INDIRECT", "INDIRECT_NOT_REVIEW"}
NOT_ACCURATE_FINE = {
    "CONTRADICT",
    "NOT_SUBSTANTIATE",
    "OVERSIMPLIFY",
    "MISQUOTE",
    "ETIQUETTE",
    # Already-collapsed label, accepted when reading data_train/.
    "NOT_ACCURATE",
}
IRRELEVANT_FINE = {"IRRELEVANT"}


def fine_to_three(label):
    if label in ACCURATE_FINE:
        return "ACCURATE"
    if label in IRRELEVANT_FINE:
        return "IRRELEVANT"
    if label in NOT_ACCURATE_FINE:
        return "NOT_ACCURATE"
    raise ValueError(f"Unknown fine label: {label}")


def normalize_pred_label(label):
    # model.decode() calls class 1 "NEI"; the paper calls it IRRELEVANT.
    if label == "NEI":
        return "IRRELEVANT"
    return label


def build_annotation_labels(zip_path):
    """Map citation-context text -> set of fine-grained annotation labels."""
    text_to_labels = defaultdict(set)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if "__MACOSX" in name:
                continue
            if "/citations/" not in name or not name.endswith(".json"):
                continue
            entry = json.loads(zf.read(name))
            spans = entry.get("citation_context") or []
            if not spans:
                continue
            # Same extraction as bm25t5.ipynb::get_context.
            start = min(s["start"] for s in spans)
            end = max(s["end"] for s in spans)
            claim = entry["citing_paragraph"][start:end].strip().replace("\n", " ")
            text_to_labels[claim].add(entry["label"])
    return text_to_labels


def load_claims(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_predictions(path):
    preds = {}
    for line in open(path, encoding="utf-8"):
        entry = json.loads(line)
        evidence = entry.get("evidence", {})
        preds[int(entry["id"])] = {int(k): v for k, v in evidence.items()}
    return preds


def gold_pair_labels(claim):
    gold = {}
    for doc_id in claim["cited_doc_ids"]:
        gold[int(doc_id)] = "IRRELEVANT"
    for doc_id, entries in claim["evidence"].items():
        labels = {fine_to_three(e["label"]) for e in entries}
        if len(labels) != 1:
            raise ValueError(f"claim {claim['id']} has mixed labels {labels}")
        gold[int(doc_id)] = labels.pop()
    return gold


def aggregate_citation(pred_evidence):
    """Pick the paragraph with the highest non-IRRELEVANT probability mass."""
    best_doc = None
    best_score = -1.0
    for doc_id, pred in pred_evidence.items():
        probs = pred.get("label_probs") or []
        if len(probs) == 3:
            # label order is [NOT_ACCURATE, NEI/IRRELEVANT, ACCURATE]
            score = float(probs[0]) + float(probs[2])
        else:
            score = 0.0 if normalize_pred_label(pred["label"]) == "IRRELEVANT" else 1.0
        if score > best_score:
            best_score = score
            best_doc = doc_id
    if best_doc is None:
        return "IRRELEVANT"
    return normalize_pred_label(pred_evidence[best_doc]["label"])


def compute_metrics(gold, pred):
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    total = len(gold)
    correct = sum(confusion[l][l] for l in LABELS)
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)
    micro_f1 = correct / total if total else 0.0  # equals accuracy for 1 label/item
    return {
        "num_examples": total,
        "accuracy": micro_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--claims",
        default=str(CODE_DIR / "data_train/target/scifact_10/claims_test.jsonl"),
    )
    parser.add_argument(
        "--annotations", default=str(REPO_ROOT / "Data" / "annotations.zip")
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    claims = load_claims(args.claims)
    preds = load_predictions(args.predictions)
    ann = build_annotation_labels(args.annotations)

    # ---------------- pair level ----------------
    pair_gold, pair_pred = [], []
    missing_pairs = 0
    for claim in claims:
        gold = gold_pair_labels(claim)
        pred_evidence = preds.get(int(claim["id"]), {})
        for doc_id, g in gold.items():
            if doc_id not in pred_evidence:
                missing_pairs += 1
                continue
            pair_gold.append(g)
            pair_pred.append(normalize_pred_label(pred_evidence[doc_id]["label"]))
    pair_metrics = compute_metrics(pair_gold, pair_pred)

    # ---------------- citation level ----------------
    cit_gold, cit_pred = [], []
    ambiguous = 0
    missing_claims = 0
    fine_breakdown = defaultdict(Counter)
    for claim in claims:
        labels = ann.get(claim["claim"], set())
        if len(labels) != 1:
            # 0: no match (should not happen); >1: same text, conflicting labels
            ambiguous += 1
            continue
        fine_label = next(iter(labels))
        gold = fine_to_three(fine_label)
        pred_evidence = preds.get(int(claim["id"]), {})
        if not pred_evidence:
            missing_claims += 1
            pred = "IRRELEVANT"
        else:
            pred = aggregate_citation(pred_evidence)
        cit_gold.append(gold)
        cit_pred.append(pred)
        fine_breakdown[fine_label][pred] += 1
    cit_metrics = compute_metrics(cit_gold, cit_pred)

    results = {
        "pair_level": pair_metrics,
        "citation_level": cit_metrics,
        "diagnostics": {
            "missing_pair_predictions": missing_pairs,
            "ambiguous_claim_texts": ambiguous,
            "claims_without_predictions": missing_claims,
        },
        "fine_grained_breakdown": {
            k: dict(v) for k, v in fine_breakdown.items()
        },
    }

    def print_metrics(name, m):
        print(f"\n== {name}: n={m['num_examples']} ==")
        print(f"micro-F1 (=accuracy): {m['micro_f1']:.4f}")
        print(f"macro-F1:             {m['macro_f1']:.4f}")
        for label in LABELS:
            c = m["per_class"][label]
            print(
                f"  {label:14s} P={c['precision']:.3f} "
                f"R={c['recall']:.3f} F1={c['f1']:.3f} n={c['support']}"
            )

    print_metrics("pair level", pair_metrics)
    print_metrics("citation level", cit_metrics)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
