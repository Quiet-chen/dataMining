#!/usr/bin/env python3
"""
Convert the released annotation data (Data/multivers-format) into the layout and
label space expected by the MultiVerS training code.

The released claims-*.jsonl files use the paper's fine-grained labels
(ACCURATE / INDIRECT / CONTRADICT / ...).  data_verisci.make_label() and
SciFactDataset only accept the three MultiVerS labels, so we collapse them
following Section 2.5.3 of the paper:

    ACCURATE, INDIRECT, INDIRECT_NOT_REVIEW          -> ACCURATE      (class 2)
    CONTRADICT, NOT_SUBSTANTIATE, OVERSIMPLIFY,
    MISQUOTE, ETIQUETTE                              -> NOT_ACCURATE  (class 0)
    IRRELEVANT                                       -> IRRELEVANT    (class 1)

"IRRELEVANT" citations have no evidence segment, so their evidence entry is
dropped and all paragraphs of the reference become class-1 candidates, which is
what the paragraph-level data module already does for every non-evidence
paragraph.

Output:
    data_train/target/scifact_10/
        claims_train.jsonl  claims_dev.jsonl  claims_test.jsonl  corpus.jsonl
    data_train/target/scifact_10/label_report.json
"""

import json
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
SRC_DIR = REPO_ROOT / "Data" / "multivers-format"
DST_DIR = CODE_DIR / "data_train" / "target" / "scifact_10"

ACCURATE_FINE = {"ACCURATE", "INDIRECT", "INDIRECT_NOT_REVIEW"}
NOT_ACCURATE_FINE = {
    "CONTRADICT",
    "NOT_SUBSTANTIATE",
    "OVERSIMPLIFY",
    "MISQUOTE",
    "ETIQUETTE",
}
IRRELEVANT_FINE = {"IRRELEVANT"}


def collapse(evidences):
    """Collapse fine-grained evidence labels to the 3-class space."""
    out = {}
    for doc_id, entries in evidences.items():
        kept = []
        for entry in entries:
            label = entry["label"]
            if label in ACCURATE_FINE:
                new_label = "ACCURATE"
            elif label in NOT_ACCURATE_FINE:
                new_label = "NOT_ACCURATE"
            elif label in IRRELEVANT_FINE:
                # No evidence segment: make the document a class-1 candidate.
                continue
            else:
                raise ValueError(f"Unknown label: {label}")
            kept.append({"sentences": entry["sentences"], "label": new_label})
        if kept:
            out[doc_id] = kept
    return out


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Corpus is shared by all splits and is already in MultiVerS format.
    corpus = {}
    for line in open(SRC_DIR / "corpus.jsonl", encoding="utf-8"):
        doc = json.loads(line)
        corpus[int(doc["doc_id"])] = doc

    report = {"num_docs": len(corpus), "splits": {}}

    for split in ["train", "dev", "test"]:
        src = SRC_DIR / f"claims-{split}.jsonl"
        dst = DST_DIR / f"claims_{split}.jsonl"
        n_claims = 0
        n_pairs = 0
        fine_counts = Counter()
        collapsed_counts = Counter()

        with open(dst, "w", encoding="utf-8") as fout:
            for line in open(src, encoding="utf-8"):
                claim = json.loads(line)
                n_claims += 1

                for entries in claim["evidence"].values():
                    for entry in entries:
                        fine_counts[entry["label"]] += 1

                claim["evidence"] = collapse(claim["evidence"])

                for entries in claim["evidence"].values():
                    for entry in entries:
                        collapsed_counts[entry["label"]] += 1

                # Validate that every cited paragraph exists and that every
                # rationale sentence index is in range.
                for doc_id in claim["cited_doc_ids"]:
                    doc_id = int(doc_id)
                    assert doc_id in corpus, f"missing doc {doc_id}"
                    n_pairs += 1
                for doc_id, entries in claim["evidence"].items():
                    doc = corpus[int(doc_id)]
                    for entry in entries:
                        for sent_idx in entry["sentences"]:
                            assert 0 <= sent_idx < len(doc["abstract"]), (
                                f"{split} claim {claim['id']} doc {doc_id} "
                                f"bad sentence index {sent_idx}"
                            )

                fout.write(json.dumps(claim, ensure_ascii=False) + "\n")

        report["splits"][split] = {
            "num_claims": n_claims,
            "num_claim_doc_pairs": n_pairs,
            "fine_label_counts": dict(fine_counts),
            "collapsed_label_counts": dict(collapsed_counts),
        }
        print(
            f"{split}: {n_claims} claims, {n_pairs} claim-doc pairs, "
            f"labels={dict(collapsed_counts)}"
        )

    # Copy the corpus under the name expected by SciFactReader.
    with open(DST_DIR / "corpus.jsonl", "w", encoding="utf-8") as fout:
        for doc_id in sorted(corpus):
            fout.write(json.dumps(corpus[doc_id], ensure_ascii=False) + "\n")

    with open(DST_DIR / "label_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {DST_DIR}")


if __name__ == "__main__":
    main()
