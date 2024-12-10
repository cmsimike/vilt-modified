from categories import get_categories
from dotenv import load_dotenv
import csv
import os
from config import Config
from multi_label_stratified import multi_label_stratified_split

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import ast
import matplotlib.pyplot as plt
import seaborn as sns


load_dotenv()


def analyze_multilabel_predictions(
    csv_path, true_col="true_label", pred_col="predicted_label"
):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert string representations of lists to actual lists
    df[true_col] = df[true_col].apply(ast.literal_eval)
    df[pred_col] = df[pred_col].apply(ast.literal_eval)

    # Get only the labels that appear in the true labels
    true_labels = set()
    for labels in df[true_col].values:
        true_labels.update(labels)
    true_labels = sorted(list(true_labels))

    # Filter predicted labels to only include those that appear in true_labels
    df[pred_col] = df[pred_col].apply(
        lambda x: [label for label in x if label in true_labels]
    )

    # Convert lists to binary format using only true labels
    mlb = MultiLabelBinarizer(classes=true_labels)
    y_true_binary = mlb.fit_transform(df[true_col])
    y_pred_binary = mlb.transform(df[pred_col])

    # Calculate metrics
    results = {
        "overall_f1_micro": f1_score(y_true_binary, y_pred_binary, average="micro"),
        "overall_f1_macro": f1_score(y_true_binary, y_pred_binary, average="macro"),
        "overall_precision_micro": precision_score(
            y_true_binary, y_pred_binary, average="micro"
        ),
        "overall_recall_micro": recall_score(
            y_true_binary, y_pred_binary, average="micro"
        ),
        "per_label_metrics": {},
    }

    # Calculate per-label metrics
    for i, label in enumerate(true_labels):
        results["per_label_metrics"][label] = {
            "f1": f1_score(y_true_binary[:, i], y_pred_binary[:, i]),
            "precision": precision_score(y_true_binary[:, i], y_pred_binary[:, i]),
            "recall": recall_score(y_true_binary[:, i], y_pred_binary[:, i]),
            "support": np.sum(y_true_binary[:, i]),
        }

    return results


def print_analysis(results):
    """
    Print the analysis results in a readable format.
    """
    print("\n=== Overall Performance ===")
    print(f"Micro-averaged F1 Score: {results['overall_f1_micro']:.4f}")
    print(f"Macro-averaged F1 Score: {results['overall_f1_macro']:.4f}")
    print(f"Micro-averaged Precision: {results['overall_precision_micro']:.4f}")
    print(f"Micro-averaged Recall: {results['overall_recall_micro']:.4f}")

    print("\n=== Per-Label Performance ===")
    for label, metrics in results["per_label_metrics"].items():
        print(f"\nLabel: {label}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Support: {metrics['support']}")


if __name__ == "__main__":
    categories = get_categories()
    config = Config()

    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    result_file = os.path.join(base_directory, "openai_results.csv")

    # Analyze the predictions
    analysis_results = analyze_multilabel_predictions(
        result_file, "actual_labels", "predicted_labels"
    )
    print_analysis(analysis_results)
