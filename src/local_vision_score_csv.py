from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from categories import get_categories
from dotenv import load_dotenv
import os
import csv
from sklearn.metrics import classification_report

load_dotenv()


def evaluate(all_categories, true_labels, pred_labels):
    # Binarize the labels
    mlb = MultiLabelBinarizer(classes=all_categories)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    # Compute precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # print(classification_report(y_true, y_pred, target_names=all_categories))


def get_rows_from_csv(result_file):
    true_labels = []
    pred_labels = []

    with open(result_file, "r") as f:
        next(f)  # skip header
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows:
            true_labels.append(eval(row[1]))
            pred_labels.append(row[2].split(","))

    return true_labels, pred_labels


if __name__ == "__main__":
    csv_file_names = [
        "llama-11b_results.csv",
        "Molmo-7b_results.csv",
        "qwen2-7b_results.csv",
    ]
    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    all_categories = get_categories()
    for name in csv_file_names:
        result_file = os.path.join(base_directory, name)

        # load csv
        # parse into two lists of lists:
        ground_truth, prediction = get_rows_from_csv(result_file)

        categories = get_categories()
        results = {
            "overall_f1_micro": 0.0,
            "overall_f1_macro": 0.0,
            "overall_precision_micro": 0.0,
            "overall_recall_micro": 0.0,
            "per_label_metrics": {},
        }
        print("Evaluating", name)

        cleaned_predictions = []
        # strip whitespace in prediction rows:
        for row in prediction:
            new_p = []
            for p in row:
                p = p.strip()
                new_p.append(p)
            cleaned_predictions.append(new_p)

        filtered_predictions = []
        for row in cleaned_predictions:
            # Filter out predictions that are not in the categories
            new_p = []
            for p in row:
                if p in categories:
                    new_p.append(p)
            filtered_predictions.append(new_p)

        print("Cleaned predictions:")
        evaluate(all_categories, ground_truth, cleaned_predictions)
        print("\nFiltered predictions:")
        evaluate(all_categories, ground_truth, filtered_predictions)
        print("\n")
