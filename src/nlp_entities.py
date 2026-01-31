from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


DIAGNOSIS_TERMS = ["pneumonia", "infection", "blood pressure", "relapse", "post-surgery"]
TREATMENT_TERMS = ["antibiotics", "medication", "treatment", "switched", "hydration", "rest"]
FOLLOWUP_TERMS = ["follow-up", "recommend follow-up", "scheduled", "monitoring", "advised"]
SYMPTOM_TERMS = ["discomfort", "reaction", "pain", "symptoms"]
MEDICATION_TERMS = ["antibiotics", "medication"]


def normalize_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_terms(text: str, terms: List[str]) -> List[str]:
    text_l = text.lower()
    found = []
    for term in terms:
        if term.lower() in text_l:
            found.append(term)

    seen = set()
    out = []
    for x in found:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out


def extract_entities(note: str) -> Dict[str, List[str]]:
    note = normalize_text(note)
    return {
        "diagnosis": find_terms(note, DIAGNOSIS_TERMS),
        "treatment": find_terms(note, TREATMENT_TERMS),
        "symptoms": find_terms(note, SYMPTOM_TERMS),
        "medications": find_terms(note, MEDICATION_TERMS),
        "follow_up": find_terms(note, FOLLOWUP_TERMS),
    }


def main(data_path: str, output_path: str) -> None:
    print("\n>>> STARTING src.nlp_entities main() <<<", flush=True)
    print(f"Data path: {data_path}", flush=True)
    print(f"Output path: {output_path}\n", flush=True)

    df = pd.read_excel(data_path, engine="openpyxl")

    if "discharge_note" not in df.columns:
        raise ValueError("Expected discharge_note column in dataset")
    if "patient_id" not in df.columns:
        raise ValueError("Expected patient_id column in dataset")

    results = []
    for _, row in df.iterrows():
        results.append(
            {
                "patient_id": int(row["patient_id"]),
                "entities": extract_entities(row["discharge_note"]),
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved extracted entities to: {out_path}", flush=True)
    print(">>> FINISHED src.nlp_entities main() <<<\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="Data/Assignment_Data.xlsx")
    parser.add_argument("--output-path", type=str, default="reports/nlp_entities.json")
    args = parser.parse_args()
    main(args.data_path, args.output_path)
