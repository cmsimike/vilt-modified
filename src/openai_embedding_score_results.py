from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
from dotenv import load_dotenv
from time import time
import mlflow

load_dotenv()


class TransformerEmbeddingComparison:
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling_strategy="cls",
        similarity_threshold=0.9,
    ):
        """
        Initialize with chosen transformer model and settings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.similarity_threshold = similarity_threshold

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Using model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Pooling strategy: {pooling_strategy}")

    def pool_embeddings(self, token_embeddings, attention_mask):
        """Apply the selected pooling strategy to token embeddings."""
        if self.pooling_strategy == "cls":
            return token_embeddings[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Create mask for padded tokens
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # Sum all token embeddings with attention mask applied
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            # Count number of non-padded tokens
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            # Calculate mean
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            # Create mask for padded tokens
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # Set padded tokens to large negative value
            token_embeddings[mask == 0] = -1e9
            # Take max over tokens
            return torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def get_embedding(self, text):
        """Generate embedding for a single text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.pool_embeddings(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

        return embeddings.cpu().numpy()[0]

    def embed_batch(self, texts, batch_size=32):
        """Generate embeddings for a list of texts in batches."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = self.pool_embeddings(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings) if embeddings else np.array([])

    def find_matching_labels(self, true_labels, pred_labels):
        """
        Find true labels that have a matching prediction based on embedding similarity.
        Ensures each prediction is only matched once.

        Returns:
            tuple containing:
            - set of matched true labels
            - list of (true_label, predicted_label) pairs that matched
        """
        if not true_labels or not pred_labels:
            return set(), []

        # Generate embeddings in batches
        true_embeddings = self.embed_batch(true_labels)
        pred_embeddings = self.embed_batch(pred_labels)

        # Calculate similarities
        similarities = cosine_similarity(true_embeddings, pred_embeddings)

        # Track matched labels and their pairs
        matched_true_labels = set()
        matched_pairs = []
        used_pred_indices = set()

        # Sort all similarities in descending order with their indices
        similarity_entries = []
        for i in range(len(true_labels)):
            for j in range(len(pred_labels)):
                similarity_entries.append((similarities[i][j], i, j))

        # Sort by similarity score in descending order
        similarity_entries.sort(reverse=True)

        # Match labels greedily from highest similarity to lowest
        for similarity, true_idx, pred_idx in similarity_entries:
            # Skip if similarity is below threshold or if either label is already matched
            if (
                similarity < self.similarity_threshold
                or true_labels[true_idx] in matched_true_labels
                or pred_idx in used_pred_indices
            ):
                continue

            # Record the match
            matched_true_labels.add(true_labels[true_idx])
            used_pred_indices.add(pred_idx)
            matched_pairs.append((true_labels[true_idx], pred_labels[pred_idx]))

        return matched_true_labels, matched_pairs

    def analyze_csv(
        self,
        csv_path,
        true_col="actual_labels",
        pred_col="predicted_labels",
    ):
        """Analyze predictions using embedding similarity."""
        start_time = time()

        # Read CSV
        df = pd.read_csv(csv_path)
        df[true_col] = df[true_col].apply(ast.literal_eval)
        df[pred_col] = df[pred_col].apply(ast.literal_eval)

        results = []
        for idx, row in df.iterrows():
            true_labels = row[true_col]
            pred_labels = row[pred_col]

            matched_labels, matched_pairs = self.find_matching_labels(
                true_labels, pred_labels
            )
            match_percentage = (
                len(matched_labels) / len(true_labels) if true_labels else 0
            )

            # Calculate similarities for all pairs for visualization
            similarities = (
                cosine_similarity(
                    self.embed_batch(true_labels), self.embed_batch(pred_labels)
                ).tolist()
                if true_labels and pred_labels
                else []
            )

            results.append(
                {
                    "row": idx + 1,
                    "true_labels": true_labels,
                    "predicted_labels": pred_labels,
                    "matched_labels": list(matched_labels),
                    "matched_pairs": matched_pairs,
                    "match_percentage": match_percentage * 100,
                    "embedding_similarities": similarities,
                }
            )

        end_time = time()

        return {
            "results": results,
            "model_name": self.model_name,
            "pooling_strategy": self.pooling_strategy,
            "processing_time": end_time - start_time,
        }


def calculate_overall_accuracy(analysis_results):
    """Calculate accuracy metrics from analysis results."""
    results_list = analysis_results["results"]

    if not results_list:
        return {
            "model_name": analysis_results["model_name"],
            "pooling_strategy": analysis_results["pooling_strategy"],
            "processing_time": analysis_results["processing_time"],
            "mean_accuracy": 0,
            "weighted_accuracy": 0,
            "perfect_match_rate": 0,
            "above_threshold_rate": 0,
        }

    mean_accuracy = sum(r["match_percentage"] for r in results_list) / len(results_list)

    total_weight = sum(len(r["true_labels"]) for r in results_list)
    weighted_accuracy = (
        sum(r["match_percentage"] * len(r["true_labels"]) for r in results_list)
        / total_weight
        if total_weight > 0
        else 0
    )

    perfect_matches = sum(1 for r in results_list if r["match_percentage"] == 100)
    perfect_match_rate = (perfect_matches / len(results_list)) * 100

    good_matches = sum(1 for r in results_list if r["match_percentage"] >= 80)
    above_threshold_rate = (good_matches / len(results_list)) * 100

    return {
        "model_name": analysis_results["model_name"],
        "pooling_strategy": analysis_results["pooling_strategy"],
        "processing_time": analysis_results["processing_time"],
        "mean_accuracy": mean_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "perfect_match_rate": perfect_match_rate,
        "above_threshold_rate": above_threshold_rate,
    }


def print_accuracy_metrics(metrics):
    """Print accuracy metrics in a readable format."""
    print(f"\n=== Results for {metrics['model_name']} ===")
    print(f"Pooling Strategy: {metrics['pooling_strategy']}")
    print(f"Processing Time: {metrics['processing_time']:.2f} seconds")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.2f}%")
    print(f"Weighted Accuracy: {metrics['weighted_accuracy']:.2f}%")
    print(f"Perfect Match Rate: {metrics['perfect_match_rate']:.2f}%")
    print(f"Above 80% Match Rate: {metrics['above_threshold_rate']:.2f}%")


def print_results_as_artifact(results):
    from io import StringIO

    output = StringIO()
    output.write("\n=== Embedding-based Label Matching Analysis ===\n")
    for result in results:
        output.write(f"\nRow {result['row']}:\n")
        output.write(f"True Labels: {', '.join(result['true_labels'])}\n")
        output.write(f"Predicted Labels: {', '.join(result['predicted_labels'])}\n")
        output.write(f"Matched Labels: {', '.join(result['matched_labels'])}\n")
        for true_label, pred_label in result["matched_pairs"]:
            output.write(f"  {true_label} -> {pred_label}\n")
        output.write(f"Match Percentage: {result['match_percentage']:.2f}%\n")

        # Format the embedding similarities as a DataFrame
        similarities = result["embedding_similarities"]
        true_labels = result["true_labels"]
        pred_labels = result["predicted_labels"]

        if similarities and true_labels and pred_labels:
            # Create a DataFrame with true labels as index and predicted labels as columns
            df_sim = pd.DataFrame(similarities, index=true_labels, columns=pred_labels)
            # Limit decimal places for readability
            df_sim = df_sim.round(3)
            output.write("Embedding Similarities:\n")
            output.write(df_sim.to_string())
            output.write("\n")
        else:
            output.write("Embedding Similarities: None\n")

    output_content = output.getvalue()

    # Log the content as an artifact with MLflow
    mlflow.log_text(output_content, artifact_file="label_matching_analysis.txt")


def compare_models(csv_path, models, pooling_strategies=None):
    """Compare multiple models and pooling strategies."""
    if pooling_strategies is None:
        pooling_strategies = ["cls"]

    results = []

    for model_name in models:
        for pooling_strategy in pooling_strategies:
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("pooling_strategy", pooling_strategy)

                print(f"\nTesting {model_name} with {pooling_strategy} pooling...")

                comparison = TransformerEmbeddingComparison(
                    model_name=model_name,
                    pooling_strategy=pooling_strategy,
                    similarity_threshold=0.98,
                )
                mlflow.log_param(
                    "similarity_threshold", comparison.similarity_threshold
                )

                analysis = comparison.analyze_csv(csv_path)
                metrics = calculate_overall_accuracy(analysis)
                results.append(metrics)

                # print_accuracy_metrics(metrics)
                mlflow.log_metric("mean_accuracy", metrics["mean_accuracy"])
                mlflow.log_metric("weighted_accuracy", metrics["weighted_accuracy"])
                mlflow.log_metric("perfect_match_rate", metrics["perfect_match_rate"])
                mlflow.log_metric(
                    "above_threshold_rate", metrics["above_threshold_rate"]
                )
                mlflow.log_metric("processing_time", metrics["processing_time"])
                print_results_as_artifact(analysis["results"])

                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


def main():
    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    result_file = os.path.join(base_directory, "openai_results.csv")

    # models_to_compare = [
    #     "distilbert-base-uncased",
    #     "bert-base-uncased",
    #     "bert-large-uncased",
    # ]
    models_to_compare = [
        # Original models
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-v3-base",
        # Models potentially good for food-related semantics
        "sentence-transformers/all-MiniLM-L6-v2",  # Good for semantic similarity
        "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Compact and efficient
        "sentence-transformers/all-mpnet-base-v2",  # High performance
        # Multilingual models that might capture food terms across languages
        "xlm-roberta-base",
        "microsoft/mdeberta-v3-base",
        # Additional diverse models
        "roberta-large",
        "bert-large-uncased",
        "microsoft/deberta-v3-large",
        # Some additional transformer models
        "albert-base-v2",
        "xlnet-base-cased",
        "google/mobilebert-uncased",
        "microsoft/deberta-v3-xsmall",
    ]
    pooling_strategies = ["cls", "mean", "max"]
    with mlflow.start_run():
        results = compare_models(result_file, models_to_compare, pooling_strategies)


if __name__ == "__main__":
    # Set the tracking uri if it's set in the environment
    if os.getenv("MLFLOW_TRACKING_SERVER"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_SERVER"))

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    main()
