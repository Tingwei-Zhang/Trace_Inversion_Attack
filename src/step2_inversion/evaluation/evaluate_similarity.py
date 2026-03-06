#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm


class SimilarityMetrics:
    def __init__(self, tokenizer_name: Optional[str] = None):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @staticmethod
    def _token_f1(prediction: str, reference: str) -> float:
        import nltk

        pred_words = set(nltk.tokenize.word_tokenize(prediction))
        ref_words = set(nltk.tokenize.word_tokenize(reference))
        tp = len(pred_words & ref_words)
        precision = tp / (len(pred_words) + 1e-20)
        recall = tp / (len(ref_words) + 1e-20)
        return (2 * precision * recall) / (precision + recall + 1e-20)

    @staticmethod
    def _exact_match(prediction: str, reference: str) -> float:
        return 1.0 if prediction.strip() == reference.strip() else 0.0

    def calculate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if not predictions or not references or len(predictions) != len(references):
            raise ValueError("Predictions and references must be non-empty and same length.")

        bleu_scores = []
        token_f1_scores = []
        exact_match_scores = []
        rouge1, rouge2, rougeL = [], [], []

        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Calculating metrics"):
            pred = str(pred).strip()
            ref = str(ref).strip()

            try:
                import nltk

                pred_tokens = nltk.word_tokenize(pred)
                ref_tokens = nltk.word_tokenize(ref)
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3)
            except Exception:
                bleu = 0.0
            bleu_scores.append(bleu * 100)

            token_f1_scores.append(self._token_f1(pred, ref) * 100)
            exact_match_scores.append(self._exact_match(pred, ref) * 100)

            rouge_result = self.rouge.score(pred, ref)
            rouge1.append(rouge_result["rouge1"].fmeasure * 100)
            rouge2.append(rouge_result["rouge2"].fmeasure * 100)
            rougeL.append(rouge_result["rougeL"].fmeasure * 100)

        if self.tokenizer:
            avg_pred_token_length = np.mean(
                [len(self.tokenizer.encode(str(pred), truncation=False, add_special_tokens=True)) for pred in predictions]
            )
            avg_ref_token_length = np.mean(
                [len(self.tokenizer.encode(str(ref), truncation=False, add_special_tokens=True)) for ref in references]
            )
        else:
            avg_pred_token_length = np.mean([len(str(pred).split()) for pred in predictions])
            avg_ref_token_length = np.mean([len(str(ref).split()) for ref in references])

        return {
            "bleu": round(float(np.mean(bleu_scores)), 4),
            "token_f1": round(float(np.mean(token_f1_scores)), 4),
            "exact_match": round(float(np.mean(exact_match_scores)), 4),
            "rouge1": round(float(np.mean(rouge1)), 4),
            "rouge2": round(float(np.mean(rouge2)), 4),
            "rougeL": round(float(np.mean(rougeL)), 4),
            "num_examples": len(predictions),
            "avg_pred_token_length": round(float(avg_pred_token_length), 2),
            "avg_ref_token_length": round(float(avg_ref_token_length), 2),
        }


def load_prediction_label_pairs(jsonl_path: str):
    predictions, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            predictions.append(item["predict"])
            labels.append(item["label"])
    return predictions, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction-label similarity metrics from a JSONL file.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="JSONL with 'predict' and 'label' fields")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer for token-length stats",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="similarity_metrics.json",
        help="Output metrics JSON filename in same directory as input JSONL",
    )
    args = parser.parse_args()

    predictions, labels = load_prediction_label_pairs(args.jsonl_path)
    calculator = SimilarityMetrics(tokenizer_name=args.tokenizer_name)
    metrics = calculator.calculate(predictions, labels)

    print(f"----- Metrics for {args.jsonl_path} -----")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_path = os.path.join(os.path.dirname(args.jsonl_path), args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
