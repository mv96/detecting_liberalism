import pandas as pd
import json
from typing import Tuple, List, Dict
from pathlib import Path
from fix_json_load import LabelProcessor
from functools import reduce


class JSONLabelParser:
    def __init__(self, csv_path: str, json_paths: Dict[str, str]):
        self.csv_path = Path(csv_path)
        self.json_paths = {k: Path(p) for k, p in json_paths.items()}
        self.examples_df = None
        self.label_dfs = {}

    def _parse_json_content(self, content: str, line_num: int) -> dict:
        """Parse and clean JSON content from model response."""
        content = content.replace("```json\n", "").replace("\n```", "").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._apply_json_fixes(content, line_num)

    def _apply_json_fixes(self, content: str, line_num: int) -> dict:
        """Apply various fixes to malformed JSON content."""
        # Remove duplicate "reasoning" lines (keep the longer one)
        fixed_content = "\n".join(
            line
            for line in content.splitlines()
            if not ('"reasoning": "' in line and len(line) < 100)
        )

        # Apply JSON formatting fixes
        fixes = {
            '"}': '"}',
            '",}': '"}',
            '", }': '"}',
            ", }": "}",
            ",}": "}",
            ",]": "]",
            '"}.': '"}',
            '",\n': '"\n',
        }

        for old, new in fixes.items():
            fixed_content = fixed_content.replace(old, new)

        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError as je:
            print(f"Line {line_num}: JSON parsing error - {je}")
            print(f"Problematic content:\n{content}")
            return {}

    def _process_labels(self, labels: List[dict]) -> dict:
        """Process labels into a row with combined reasoning."""
        row = {}
        all_reasoning = []

        for item in labels:
            category = item["category"]
            row[category] = item["label"]
            all_reasoning.append(f"{category}: {item['reasoning']}")

        row["combined_reasoning"] = "\n".join(all_reasoning)
        return row

    def _load_jsonl(self, json_path: Path) -> pd.DataFrame:
        """Load and process JSONL file into DataFrame."""
        rows = []

        with open(json_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if not self._is_valid_response(data):
                    print(f"Line {i}: Invalid response structure")
                    break

                content = data["response"]["body"]["choices"][0]["message"]["content"]
                labels = self._parse_json_content(content, i)

                # Extract custom_id and transform it to just the number
                custom_id = data.get("custom_id", "")
                id_number = None
                if custom_id and custom_id.startswith("request-"):
                    try:
                        id_number = int(custom_id.split("request-")[1])
                    except (ValueError, IndexError):
                        print(f"Line {i}: Invalid custom_id format: {custom_id}")

                if labels:
                    row = self._process_labels(labels)
                    row["index"] = id_number  # Change from "id" to "index"
                    rows.append(row)

        df = pd.DataFrame(rows)

        def convert_to_int(value):
            if isinstance(value, str):
                value = value.strip().lower()
                if value == "yes":
                    return 1
                elif value == "no":
                    return 0
            return value

        label_cols = [col for col in df.columns if "label" in col]
        for col in label_cols:
            df[col] = df[col].apply(convert_to_int)

        return df

    def _is_valid_response(self, data: dict) -> bool:
        """Check if response structure is valid."""
        return (
            "response" in data
            and "body" in data["response"]
            and "choices" in data["response"]["body"]
            and len(data["response"]["body"]["choices"]) > 0
            and "message" in data["response"]["body"]["choices"][0]
            and "content" in data["response"]["body"]["choices"][0]["message"]
        )

    def load_data(self) -> None:
        """Load all data sources."""
        self.examples_df = pd.read_csv(self.csv_path)

        for model_name, json_path in self.json_paths.items():
            try:
                df = self._load_jsonl(json_path)
            except:
                processor = LabelProcessor()
                result = processor.load_json(json_path)
                # Check if result is a tuple and extract the DataFrame
                if isinstance(result, tuple):
                    df = result[0]  # Assuming the DataFrame is the first element
                else:
                    df = result

            self.label_dfs[model_name] = df

    def merge_and_save(self, output_path: str) -> pd.DataFrame:
        """Merge all DataFrames and save to CSV."""
        if self.examples_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        final_df = None
        renamed_dfs = []
        for model_name, df in self.label_dfs.items():
            # Rename columns to add model name suffix except for 'index'
            renamed_df = df.copy()
            columns_to_rename = {
                col: f"{col}_{model_name}" for col in df.columns if col != "index"
            }
            renamed_df = renamed_df.rename(columns=columns_to_rename)

            renamed_dfs.append(renamed_df)
            print(renamed_df.shape)

        print("=" * 100)

        # Merge all dataframes in one step but inner join
        final_df = reduce(
            lambda left, right: pd.merge(left, right, on="index", how="inner"),
            renamed_dfs,
        )

        rows_to_filter = final_df["index"]

        # filter the rows from examples_df that are  in rows_to_filter as the index
        examples_df_filtered = self.examples_df[
            self.examples_df.index.isin(rows_to_filter)
        ]

        # concat the filtered examples_df with the final_df
        print(examples_df_filtered.shape, final_df.shape)

        # Reset indexes on both dataframes to ensure alignment
        examples_df_filtered = examples_df_filtered.reset_index(drop=True)
        final_df = final_df.reset_index(drop=True)

        # Now concatenate horizontally
        final_df = pd.concat([examples_df_filtered, final_df], axis=1)

        # select all the rows from example_df using the list rows_to_filter
        final_df = pd.concat(
            [
                final_df,
                self.examples_df[~self.examples_df.index.isin(rows_to_filter)],
            ]
        )

        print(final_df.shape)

        # drop the id column and rearrange the column so the first column is the index the second is the translation and the last is every thing else
        final_df = final_df.drop(columns=["id"])
        final_df = final_df[
            [
                "index",
                "translation",
                *[
                    col
                    for col in final_df.columns
                    if col != "index" and col != "translation"
                ],
            ]
        ]

        # Rearrange the first column as index column without resetting the index

        # Remove rows with NaN values
        initial_rows = len(final_df)
        final_df = final_df.dropna()
        rows_removed = initial_rows - len(final_df)

        print(f"\nRows before filtering: {initial_rows}")
        print(f"Rows after filtering: {len(final_df)}")
        print(f"Rows removed: {rows_removed}")

        # Find all columns with "label" in their name
        label_columns = [col for col in final_df.columns if "label" in col]
        print(f"\nFound {len(label_columns)} label columns to convert")

        # Save to CSV with numeric values
        final_df.to_csv(output_path, index=False)
        print(f"Combined data with numeric labels saved to {output_path}")

        return final_df


if __name__ == "__main__":
    # Configuration
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

    try:
        # Initialize and run the parser
        parser = JSONLabelParser(config["csv_path"], config["json_paths"])
        parser.load_data()
        final_df = parser.merge_and_save(config["output_path"])

        # Display results
        print("\nFirst few rows of the combined DataFrame:")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
