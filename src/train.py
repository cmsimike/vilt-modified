# Influenced by https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb#scrollTo=VpxDoZ9N4KyC
import os
import torch
from transformers import ViltProcessor
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from config import Config
from dataset import create_dataloaders
from model import ViltForMultiLabelClassification
import dotenv
import mlflow
from torchinfo import summary
from labels import LabelProcessor
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np
from io import StringIO

dotenv.load_dotenv()


# Get classification report and confusion matrices as a formatted string.
def get_metrics_string(all_preds, all_labels, category_names):
    output = StringIO()

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Get classification report with actual labels
    report = classification_report(
        y_true, y_pred, target_names=category_names, zero_division=0
    )

    # Get confusion matrix for each class
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    output.write("=== Classification Report ===\n")
    output.write(report)
    output.write("\n=== Confusion Matrices ===\n")

    # Print confusion matrix for each category
    for i, matrix in enumerate(confusion_matrices):
        output.write(f"\n{category_names[i]}:\n")
        output.write("TN FP\nFN TP\n")
        output.write(str(matrix) + "\n")

    return output.getvalue()


def format_predictions_to_string(all_preds, all_labels, label_processor):
    """Format predictions and ground truth into a concise string."""
    output = ["ground truth, predictions"]

    for pred, label in zip(all_preds, all_labels):
        pred_cats = [
            label_processor.idx_to_label[i] for i in range(len(pred)) if pred[i]
        ]
        true_cats = [
            label_processor.idx_to_label[i] for i in range(len(label)) if label[i]
        ]
        output.append(f"{true_cats}, {pred_cats}")

    return "\n".join(output)


def train(config):
    label_processor = LabelProcessor()
    with mlflow.start_run(nested=True):
        mlflow.log_params(config.to_dict())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        train_loader, test_loader = create_dataloaders(config, processor)

        model = ViltForMultiLabelClassification(config.NUM_LABELS)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        best_f1 = float("-inf")
        # Training loop
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            train_loss = 0

            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}"
            )
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}

                _, loss = model(**batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            mlflow.log_metric(
                "average-training-loss",
                round((train_loss / len(train_loader)), 4),
                step=(epoch + 1),
            )

            # Evaluation loop
            model.eval()
            val_f1 = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch["labels"]
                    logits, _ = model(**batch)
                    predictions = torch.sigmoid(logits) > config.FALSE_THRESHOLD
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate F1 score
            val_f1 = f1_score(
                np.array(all_labels), np.array(all_preds), average="micro"
            )
            mlflow.log_metric("validation-f1", round(val_f1, 4), step=(epoch + 1))
            # Save best model
            # TODO question - do we save based on f1 score or loss?
            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save the model when it performs better
                mlflow.pytorch.log_model(model, "model")
                prediction_results = format_predictions_to_string(
                    all_preds, all_labels, label_processor
                )
                metrics_str = get_metrics_string(
                    all_preds, all_labels, label_processor.get_categories()
                )
                mlflow.log_text(metrics_str, "metrics.txt")
                mlflow.log_text(prediction_results, "prediction_results.txt")
                mlflow.log_metric(
                    "best-validation-f1", round(best_f1, 4), step=(epoch + 1)
                )
                print(f"New best model saved with F1 score: {best_f1}")

        mlflow.log_text(str(summary(model)), "model_summary.txt")
        return model, best_f1


if __name__ == "__main__":
    # Set the tracking uri if it's set in the environment
    if os.getenv("MLFLOW_TRACKING_SERVER"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_SERVER"))

    # Fixed parameters
    FIXED_TRAIN_SPLIT = 0.8
    FIXED_RANDOM_SEED = 42

    # Lists of different parameter values to try
    batch_sizes = [4, 8, 16, 32]
    num_epochs = [30]
    learning_rates = [1e-5, 3e-5, 5e-5]
    vilt_weight_lock = [True, False]  # lock VILT weights?
    false_thresholds = [0.3, 0.5, 0.7]  # classification thresholds

    # Debugging config
    # batch_sizes = [16]
    # num_epochs = [2]
    # learning_rates = [1e-5]
    # vilt_weight_lock = [True]
    # false_thresholds = [0.5]

    with mlflow.start_run():
        # Create all combinations and evaluate
        for batch_size in batch_sizes:
            for epochs in num_epochs:
                for lr in learning_rates:
                    for lock_weights in vilt_weight_lock:
                        for threshold in false_thresholds:
                            # Create config with current parameter combination
                            config = Config(
                                batch_size=batch_size,
                                num_epochs=epochs,
                                learning_rate=lr,
                                train_split=FIXED_TRAIN_SPLIT,
                                random_seed=FIXED_RANDOM_SEED,
                                lock_vilt_weights=lock_weights,
                                false_threshold=threshold,
                            )
                            # Create the experiment if it doesn't exist
                            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
                            if mlflow.get_experiment_by_name(experiment_name) is None:
                                mlflow.create_experiment(experiment_name)
                            mlflow.set_experiment(experiment_name)

                            model, best_f1 = train(config)
