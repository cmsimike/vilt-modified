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
import pprint


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
    result_file = os.path.join(base_directory, "openai_results_extra_structure.csv")

    # Simply verify the "predicted_labels" vs actual categories
    with open(result_file, "r") as f:
        next(f)  # skip header
        reader = csv.reader(f)
        extra_categories = {}
        category_or_ingredient = {}
        for row in reader:
            ground_truth = ast.literal_eval(row[1])
            # using ast.literal_eval to convert string to json because openai gave me back single quote-parameters
            prediction_obj = ast.literal_eval(row[2])

            for category in prediction_obj:
                category_name = category["category"]
                if category_name not in categories:
                    if "category" not in category_or_ingredient:
                        category_or_ingredient["category"] = 0
                    category_or_ingredient["category"] += 1
                    if category_name not in extra_categories:
                        extra_categories[category_name] = 0
                    extra_categories[category_name] += 1

                category_ingredients = category["ingredients"]
                for ingredient in category_ingredients:
                    if ingredient not in categories:
                        if "ingredient" not in category_or_ingredient:
                            category_or_ingredient["category"] = 0
                        category_or_ingredient["category"] += 1
                        if ingredient not in extra_categories:
                            extra_categories[ingredient] = 0
                        extra_categories[ingredient] += 1
        pprint.pprint(extra_categories)
        pprint.pprint(category_or_ingredient)
# {'': 1,
#  ' ': 1,
#  ' muffin': 2,
#  'angel food cake': 1,
#  'banana bread': 1,
#  'basil': 1,
#  'bavarian cream': 1,
#  'bean sprouts': 1,
#  'beans': 1,
#  'beer': 1,
#  'birthday cake': 1,
#  'biscotti': 1,
#  'brownies': 1,
#  'bundt cake': 1,
#  'butter': 1,
#  'cakes': 1,
#  'cannoli': 1,
#  'chopsticks': 1,
#  'choux pastry': 1,
#  'cilantro': 1,
#  'clam chowder': 1,
#  'coffee cake': 1,
#  'cookies and cream': 1,
#  'cream': 1,
#  'cream puff': 1,
#  'croquembouche': 1,
#  'cucumbers': 1,
#  'cup caket': 1,
#  'cups': 1,
#  'danish': 1,
#  'dough': 1,
#  'eclairs': 1,
#  'eggs': 2,
#  'financier': 1,
#  'fork': 1,
#  'french toast': 1,
#  'fruit cake': 1,
#  'galette': 1,
#  'garlic': 1,
#  'gateau': 1,
#  'genoise': 1,
#  'glass': 1,
#  'grand marnier soufflÃ©': 1,
#  'gravy': 1,
#  'herbs': 3,
#  'ice': 1,
#  'lemon': 2,
#  'lemon cake': 1,
#  'lemons': 2,
#  'lime': 1,
#  'limes': 1,
#  'linzer torte': 1,
#  'macaron': 2,
#  'madeline': 1,
#  'mille feuille': 1,
#  'mint': 1,
#  'mozzarella': 1,
#  'napoleon cake': 1,
#  'noodles': 1,
#  'oil': 1,
#  'olive oil': 1,
#  'opera cake': 1,
#  'oranges': 1,
#  'oregano': 1,
#  'pancakes': 1,
#  'parmesan': 1,
#  'parsley': 9,
#  'pastris': 3,
#  'pastry cream': 1,
#  'peanuts': 1,
#  'pepper': 2,
#  'pepperoni': 1,
#  'pie': 1,
#  'plate': 3,
#  'pound cake': 1,
#  'profiteroles': 1,
#  'pumpkin': 1,
#  'red velvet cake': 1,
#  'rosemary': 1,
#  'sacher torte': 1,
#  'salt': 3,
#  'sausage': 1,
#  'savarin': 1,
#  'sesame': 1,
#  'shortcake': 1,
#  'soup': 2,
#  'sponge cake': 1,
#  'tart': 1,
#  'tarte tatin': 1,
#  'tiramisu': 1,
#  'torte': 1,
#  'trifle': 1,
#  'vanilla cake': 1,
#  'vegetables': 1,
#  'vienna cake': 1,
#  'wedding cake': 1,
#  'white wine': 1,
#  'wine': 2,
#  'xiao long bao': 1,
#  'xigua': 1,
#  'xocolatl': 1,
#  'yakiniku': 1,
#  'yakitori': 1,
#  'yokan': 1,
#  'yule log': 1,
#  'zabaglione': 1,
#  'zabapa': 1,
#  'zagorsky': 1,
#  'zarzuela': 1,
#  'zathura': 1,
#  'zatoba': 1,
#  'zaza': 1,
#  'zehr': 1,
#  'zeiah': 1,
#  'zeke': 1,
#  'zelda cake': 1,
#  'zelig cake': 1,
#  'zell cake': 1,
#  'zen cake': 1,
#  'zenner': 1,
#  'zephyr': 1,
#  'zeppelin': 1,
#  'zeppelin cake': 1,
#  'zeppole': 1,
#  'zerai': 1,
#  'zerberus': 1,
#  'zerbinet': 1,
#  'zerbinetta': 1,
#  'zeroth': 1,
#  'zesty lemon cake': 1,
#  'zeus': 1,
#  'zeus cake': 1,
#  'zhakhnoun': 1,
#  'ziemer': 1,
#  'ziempniak': 1,
#  'ziena': 1,
#  'ziergalla': 1,
#  'zierig': 1,
#  'zig-zag cake': 1,
#  'zigadon': 1,
#  'zigart': 1,
#  'ziggurat cake': 1,
#  'ziggy': 1,
#  'zighina': 1,
#  'zigzag': 1,
#  'zikmund': 1,
#  'zikrit': 1,
#  'zil': 1,
#  'zilfiqar': 1,
#  'zilia': 1,
#  'zilla': 1,
#  'zillig': 1,
#  'zillionaire cake': 1,
#  'zilnbrignen': 1,
#  'zima cake': 1,
#  'zimalia': 1,
#  'zimeren': 1,
#  'zimira': 1,
#  'zimm': 1,
#  'zimmerman': 1,
#  'zimnuhom': 1,
#  'zimovia': 1,
#  'zimphire': 1,
#  'zimri': 1,
#  'zinar': 1,
#  'zinaret': 1,
#  'zinchard': 1,
#  'zingarelli': 1,
#  'zingel': 1,
#  'zingler': 1,
#  'zinia': 1,
#  'zinina': 1,
#  'zinkiv': 1,
#  'zinn': 1,
#  'zinnia': 1,
#  'zinnia cake': 1,
#  'zinnie': 1,
#  'zino': 1,
#  'zinora': 1,
#  'zinsberger': 1,
#  'zinzendorf': 1,
#  'ziolkowice': 1,
#  'zion': 1,
#  'zipper cake': 1,
#  'zippy': 1,
#  'zircon': 1,
#  'zirconium': 1,
#  'zither': 1,
#  'zloty cake': 1,
#  'zo': 1,
#  'zod': 1,
#  'zoder': 1,
#  'zodiac': 1,
#  'zodiac killer': 1,
#  'zoetropes': 1,
#  'zohar': 1,
#  'zoidberg cake': 1,
#  'zoleta': 1,
#  'zollars': 1,
#  'zolotono,': 1,
#  'zombie': 1,
#  'zombie cake': 1,
#  'zonal cake': 1,
#  'zonga': 1,
#  'zoning': 1,
#  'zonk cake': 1,
#  'zoo cake': 1,
#  'zooms': 1,
#  'zoosome': 1,
#  'zootopia': 1,
#  'zor': 1,
#  'zorbike cake': 1,
#  'zot cake': 1,
#  'zounds': 1,
#  'zoysia': 1,
#  'zuberec': 1,
#  'zucchini cake': 1,
#  'zukovac': 1,
#  'zulu': 1,
#  'zusha': 1,
#  'zuvice': 1,
#  'zvezda': 1,
#  'zwergens': 1,
#  'zwick': 1,
#  'zwieback': 1,
#  'zwingli': 1,
#  'zydeco': 1,
#  'zydeco cake': 1,
#  'zypher': 1,
#  'zyskind': 1}
# {'category': 46}
# # Analyze the predictions
# analysis_results = analyze_multilabel_predictions(
#     result_file, "actual_labels", "predicted_labels"
# )
# print_analysis(analysis_results)
