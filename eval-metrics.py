import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import json
import evaluate
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_agreement(csv_path, model1_cols, model2_cols):
    """
    Calculate agreement metrics between two models' predictions across multiple labels.

    Args:
        csv_path (str): Path to the CSV file
        model1_cols (list): List of column names for the first model's predictions
        model2_cols (list): List of column names for the second model's predictions

    Returns:
        dict: Dictionary containing agreement metrics
    """
    df = pd.read_csv(csv_path)

    # Ensure all columns exist
    all_cols = model1_cols + model2_cols
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in CSV")
        return None

    total_rows = len(df)

    # Calculate full agreement (all labels match)
    full_agreements = 0
    for idx in range(total_rows):
        if all(
            df.loc[idx, col1] == df.loc[idx, col2]
            for col1, col2 in zip(model1_cols, model2_cols)
        ):
            full_agreements += 1

    # Calculate partial agreement (at least half of labels match)
    partial_agreements = 0
    min_matches_required = len(model1_cols) / 2  # At least half of labels must match

    for idx in range(total_rows):
        matching_labels = sum(
            df.loc[idx, col1] == df.loc[idx, col2]
            for col1, col2 in zip(model1_cols, model2_cols)
        )
        if matching_labels >= min_matches_required:
            partial_agreements += 1

    # Calculate total agreement (agreement ratio across all labels)
    total_agreements = 0
    total_comparisons = total_rows * len(model1_cols)

    for col1, col2 in zip(model1_cols, model2_cols):
        total_agreements += (df[col1] == df[col2]).sum()

    metrics = {
        "full_agreement": {
            "count": full_agreements,
            "percentage": (full_agreements / total_rows) * 100,
        },
        "partial_agreement": {
            "count": partial_agreements,
            "percentage": (partial_agreements / total_rows) * 100,
        },
        "total_agreement": {
            "count": total_agreements,
            "percentage": (total_agreements / total_comparisons) * 100,
        },
        "total_samples": total_rows,
        "total_comparisons": total_comparisons,
    }

    return metrics


def calculate_per_column_agreement(csv_path, model1_cols, model2_cols):
    """
    Calculate agreement metrics between two models' predictions for each column pair.

    Args:
        csv_path (str): Path to the CSV file
        model1_cols (list): List of column names for the first model's predictions
        model2_cols (list): List of column names for the second model's predictions

    Returns:
        dict: Dictionary containing per-column agreement metrics
    """
    df = pd.read_csv(csv_path)

    # Ensure all columns exist
    all_cols = model1_cols + model2_cols
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in CSV")
        return None

    total_rows = len(df)
    per_column_metrics = {}

    # Calculate agreement for each column pair
    for col1, col2 in zip(model1_cols, model2_cols):
        agreements = (df[col1] == df[col2]).sum()
        per_column_metrics[f"{col1} vs {col2}"] = {
            "count": agreements,
            "percentage": (agreements / total_rows) * 100,
        }

    return per_column_metrics


def calculate_text_metrics(csv_path, model1_reasoning_col, model2_reasoning_col):
    """
    Calculate BLEU, ROUGE-L, and BERTScore between two models' reasoning outputs.

    Args:
        csv_path (str): Path to the CSV file
        model1_reasoning_col (str): Column name for first model's reasoning
        model2_reasoning_col (str): Column name for second model's reasoning

    Returns:
        dict: Dictionary containing the calculated metrics
    """
    # Download required NLTK data
    try:
        nltk.download("punkt_tab", quiet=True)
    except:
        pass

    df = pd.read_csv(csv_path)

    # Ensure columns exist
    if model1_reasoning_col not in df.columns or model2_reasoning_col not in df.columns:
        print(f"Warning: Reasoning columns not found in CSV")
        return None

    # Initialize scorers
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bert_scorer = BERTScorer(
        model_type="distilbert-base-uncased",
        use_fast_tokenizer=True,
        lang="en",
        rescale_with_baseline=True,
        batch_size=256,  # Adjust this based on your GPU memory
    )
    smoother = SmoothingFunction().method1

    # Initialize metric aggregates
    bleu_scores = []
    rouge_l_scores = []

    print("Calculating BLEU and ROUGE-L scores...")
    # Calculate BLEU and ROUGE-L scores with progress bar
    for idx in tqdm(range(len(df)), desc="Processing texts"):
        ref = df.loc[idx, model1_reasoning_col]
        hyp = df.loc[idx, model2_reasoning_col]

        # Skip if either is NaN
        if pd.isna(ref) or pd.isna(hyp):
            continue

        # Tokenize for BLEU
        ref_tokens = nltk.word_tokenize(str(ref))
        hyp_tokens = nltk.word_tokenize(str(hyp))

        # Calculate BLEU
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
        bleu_scores.append(bleu)

        # Calculate ROUGE-L
        rouge_scores = rouge_scorer_instance.score(str(ref), str(hyp))
        rouge_l_scores.append(rouge_scores["rougeL"].fmeasure)

    print("Calculating BERTScore...")
    valid_refs = df[model1_reasoning_col].dropna().tolist()
    valid_hyps = df[model2_reasoning_col].dropna().tolist()

    if valid_refs and valid_hyps:  # Only calculate if we have valid texts
        # Process BERTScore in batches with progress bar
        batch_size = 256
        P_list = []
        R_list = []
        F1_list = []

        for i in tqdm(
            range(0, len(valid_refs), batch_size), desc="Computing BERTScore"
        ):
            batch_refs = valid_refs[i : i + batch_size]
            batch_hyps = valid_hyps[i : i + batch_size]
            P, R, F1 = bert_scorer.score(batch_hyps, batch_refs)
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)

        # Concatenate all batch results
        P = torch.cat(P_list)
        R = torch.cat(R_list)
        F1 = torch.cat(F1_list)

        bert_scores = {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    else:
        bert_scores = {"precision": 0, "recall": 0, "f1": 0}

    metrics = {
        "bleu": {"mean": np.mean(bleu_scores), "std": np.std(bleu_scores)},
        "rouge_l": {"mean": np.mean(rouge_l_scores), "std": np.std(rouge_l_scores)},
        "bert_score": bert_scores,
    }

    return metrics


def plot_f1_heatmap_all(csv_path, models_data):
    df = pd.read_csv(csv_path)  # Read the CSV file
    f1_scores = {}
    model_names = list(models_data.keys())

    # Calculate F1 scores for each model pair
    for model1 in model_names:
        y_test = df[models_data[model1]].values  # True labels
        for model2 in model_names:
            y_pred = df[models_data[model2]].values  # Predicted labels
            f1 = f1_score(
                y_test, y_pred, average="weighted"  # weighted
            )  # Use weighted average for multiclass
            f1_scores[f"{model1} vs {model2}"] = f1

    # Create a DataFrame for heatmap
    f1_matrix = np.zeros((len(model_names), len(model_names)))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            f1_matrix[i, j] = f1_scores[f"{model1} vs {model2}"]

    # Plotting the heatmap - increase size from (8, 6) to (12, 10)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".2f",  # Change from .2f to .1f for one decimal place
        cmap="YlGnBu",
        xticklabels=model_names,
        yticklabels=model_names,
    )
    plt.title("F1 Score Heatmap for All Models")
    plt.xlabel("Models")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.savefig("f1_heatmap_all_models.png")
    plt.close()


def calculate_agreement_metrics(csv_path, model_columns_dict):
    df = pd.read_csv(csv_path)
    model_names = list(model_columns_dict.keys())

    # Initialize matrices to hold metrics
    full_agreement_matrix = np.zeros((len(model_names), len(model_names)))
    partial_agreement_matrix = np.zeros((len(model_names), len(model_names)))
    total_agreement_matrix = np.zeros((len(model_names), len(model_names)))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            metrics = calculate_agreement(
                csv_path, model_columns_dict[model1], model_columns_dict[model2]
            )
            if metrics:
                full_agreement_matrix[i, j] = metrics["full_agreement"]["percentage"]
                partial_agreement_matrix[i, j] = metrics["partial_agreement"][
                    "percentage"
                ]
                total_agreement_matrix[i, j] = metrics["total_agreement"]["percentage"]

    return (
        full_agreement_matrix,
        partial_agreement_matrix,
        total_agreement_matrix,
        model_names,
    )


def plot_agreement_heatmap(
    full_agreement_matrix, partial_agreement_matrix, total_agreement_matrix, model_names
):
    # Create a combined heatmap
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Full Agreement Heatmap
    sns.heatmap(
        full_agreement_matrix,  # Keep as float for decimal display
        annot=True,
        fmt=".1f",  # Display two decimal points
        cmap="YlGnBu",
        ax=axes[0],
        xticklabels=model_names,
        yticklabels=model_names,
        annot_kws={"size": 6},  # Set font size for annotations smaller
    )
    axes[0].set_title("Full Agreement Percentage")
    axes[0].set_xlabel("Models")
    axes[0].set_ylabel("Models")

    # Partial Agreement Heatmap
    sns.heatmap(
        partial_agreement_matrix,  # Keep as float for decimal display
        annot=True,
        fmt=".2f",  # Display two decimal points
        cmap="YlGnBu",
        ax=axes[1],
        xticklabels=model_names,
        yticklabels=model_names,
        annot_kws={"size": 6},  # Set font size for annotations smaller
    )
    axes[1].set_title("Partial Agreement Percentage")
    axes[1].set_xlabel("Models")
    axes[1].set_ylabel("Models")

    # Total Agreement Heatmap
    sns.heatmap(
        total_agreement_matrix,  # Keep as float for decimal display
        annot=True,
        fmt=".2f",  # Display two decimal points
        cmap="YlGnBu",
        ax=axes[2],
        xticklabels=model_names,
        yticklabels=model_names,
        annot_kws={"size": 6},  # Set font size for annotations smaller
    )
    axes[2].set_title("Total Agreement Percentage")
    axes[2].set_xlabel("Models")
    axes[2].set_ylabel("Models")

    plt.tight_layout()
    plt.savefig("agreement_metrics_heatmap.png")
    plt.close()


# Example usage
if __name__ == "__main__":
    path_to_csv = "all_labels.csv"
    df = pd.read_csv(path_to_csv)

    config = {
        "csv_path": "701_new_examples.csv",
        "json_paths": {
            "o3": "closed_source/responses/o3-resp.jsonl",
            "4o": "closed_source/responses/gpt40-resp.jsonl",
            "deepseek-r1:1.5b": "open_models/ollama_labels_deepseek-r1:1.5b.json",
            "deepseek-r1:8b": "open_models/ollama_labels_deepseek-r1:8b.json",
            "deepseek-r1:7b": "open_models/ollama_labels_deepseek-r1.json",
            "llama3.2:1b": "open_models/ollama_labels_llama3.2:1b.json",
            "llama3.2:3b": "open_models/ollama_labels_llama3.2.json",
            "mistral:7bv0.3": "open_models/ollama_labels_mistral.json",
            "qwen2.5:7b": "open_models/ollama_labels_qwen2.5:latest.json",
            "qwen2.5:1.5b": "open_models/ollama_labels_qwen2.5:1.5b.json",
            "qwen2.5:14b": "open_models/ollama_labels_qwen2.5:14b.json",
            "qwen2.5:3b": "open_models/ollama_labels_qwen2.5:3b.json",
            "qwen2.5:0.5b": "open_models/ollama_labels_qwen2.5:0.5b.json",
            "phi4:14b": "open_models/ollama_labels_phi4:latest.json",
            "phi4:3.8b": "open_models/ollama_labels_phi4-mini:latest.json",
            "gemma3_1b": "open_models/ollama_labels_gemma3_1b.json",
            "gemma3_4b": "open_models/ollama_labels_gemma3_latest.json",
            "gemma3_12b": "open_models/ollama_labels_gemma3_12b.json",
            "gemma3_27b": "open_models/ollama_labels_gemma3_27b.json",
        },
        "output_path": "all_labels.csv",
    }

    models = list(config["json_paths"].keys())

    col_models = []
    for model in models:
        col_model = [
            col
            for col in df.columns.str.lower()
            if model in col and "reasoning" not in col
        ]
        col_models.append(sorted(col_model))

    model_columns_dict = {}
    for i, model in enumerate(models):
        model_columns_dict[f"Model {model}"] = col_models[i]

    plot_f1_heatmap_all(path_to_csv, model_columns_dict)
    # Calculate agreement metrics
    (
        full_agreement_matrix,
        partial_agreement_matrix,
        total_agreement_matrix,
        model_names,
    ) = calculate_agreement_metrics(path_to_csv, model_columns_dict)

    # Plot the agreement metrics heatmap
    plot_agreement_heatmap(
        full_agreement_matrix,
        partial_agreement_matrix,
        total_agreement_matrix,
        model_names,
    )
    exit()
    # Add after the existing metrics calculations:
    reasoning_cols = {
        "Model O3": next(
            col
            for col in df.columns
            if "o3" in col.lower() and "reasoning" in col.lower()
        ),
        "Model 4O": next(
            col
            for col in df.columns
            if "4o" in col.lower() and "reasoning" in col.lower()
        ),
    }

    text_metrics = calculate_text_metrics(
        path_to_csv, reasoning_cols["Model O3"], reasoning_cols["Model 4O"]
    )

    if text_metrics:
        print("\nText Similarity Metrics for Reasoning:")
        print(json.dumps(text_metrics, indent=2))

    # Call the new function to plot the heatmap
    plot_f1_heatmap(confusion_matrices)
    plot_f1_heatmap_all(path_to_csv, model_columns_dict)

    metrics_data = calculate_agreement_metrics(path_to_csv, model_columns_dict)

    # Plot the agreement metrics
    plot_agreement_metrics(metrics_data)
