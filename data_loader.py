from datasets import load_dataset
import pandas as pd

def load_medhallu():
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")

    data = dataset['train']

    rows = []

    for item in data:
        question = item['Question']
        ground_truth = item['Ground Truth']
        hallucinated = item['Hallucinated Answer']

        rows.append({
            "text": question + " " + ground_truth,
            "label": 0
        })

        rows.append({
            "text": question + " " + hallucinated,
            "label": 1
        })

    df = pd.DataFrame(rows)
    return df