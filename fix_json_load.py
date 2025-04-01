import json
import pandas as pd
from pathlib import Path
import os
from typing import List, Dict, Any, Optional


class LabelProcessor:
    """Class to process political stance label data from JSON files."""

    ALL_LABELS = [
        "partisan_state_label",
        "closed_society_label",
        "power_concentration_label",
        "euroscepticism_label",
        "economic_label",
        "censorship_label",
        "immigration_label",
    ]

    def extract_labels(self, response: Any) -> List[Dict[str, Any]]:
        """Extract labels from the response data.

        Args:
            response: The response data containing labels

        Returns:
            List of dictionaries with category, label, and reasoning
        """
        # Handle empty response
        if response == "":
            return [
                {"category": label, "label": 0, "reasoning": ""}
                for label in self.ALL_LABELS
            ]

        # Remove empty strings from list
        if isinstance(response, list) and "" in response:
            response = [element for element in response if element != ""][0]

        # Handle case where response is a specific structure dictionary
        if isinstance(response, dict) and len(response) == 7:
            # Case 1: Special structure with list values
            if all(isinstance(response.get(key), list) for key in response):
                return [
                    {"category": category, "label": value[2], "reasoning": value[1]}
                    for category, value in response.items()
                ]
            # Case 2: Simple dictionary case
            else:
                return [
                    {"category": cat, "label": response[cat], "reasoning": ""}
                    for cat in response
                ]

        # Handle standard list of dicts case
        if isinstance(response, list) and all(
            isinstance(item, dict) for item in response if item
        ):
            # Create mapping from category to element
            response_dict = {
                element["category"]: element
                for element in response
                if isinstance(element, dict)
            }

            return [
                {
                    "category": label,
                    "label": response_dict.get(label, {}).get("label", 0),
                    "reasoning": response_dict.get(label, {}).get("reasoning", ""),
                }
                for label in self.ALL_LABELS
            ]

        # Default case - return empty labels
        return [
            {"category": label, "label": 0, "reasoning": ""}
            for label in self.ALL_LABELS
        ]

    def convert_to_int(self, value: Any) -> int:
        """Convert a value to an integer.

        Args:
            value: The value to convert

        """

        if pd.isna(value):
            return 0

        def check_string(value):
            possible_yes = ["yes", True, "true", "y", "1"]
            possible_no = [
                "no",
                False,
                "false",
                "n",
                "0",
                "",
                "n/a",
                "not applicable",
                "unknown",
                "none",
            ]

            # Check if any 'yes' indicator exists as substring
            for yes_value in possible_yes:
                if str(yes_value) in value.lower():
                    return 1
            # Check for exact matches first
            if value in possible_yes:
                return 1
            if value in possible_no:
                return 0
            else:
                return 0

        if isinstance(value, str):
            value = value.strip().lower()
            return check_string(value)
        # check if value is nan
        if isinstance(value, (int, float)):
            if value == 1:
                return 1
            else:
                return 0

        if isinstance(value, dict):
            if "label" in value and isinstance(value["label"], str):
                return check_string(value["label"])
            else:
                return 0
        if isinstance(value, bool):
            if value == True:
                return 1
            else:
                return 0

        # worst case we return as is
        return value

    def load_json(self, json_path: Path) -> pd.DataFrame:
        """Load and process JSON file into DataFrame.

        Args:
            json_path: Path to the JSON file

        Returns:
            Processed DataFrame with labels and reasoning
        """
        all_data = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {json_path}: {e}")
            return pd.DataFrame()

        for i, item in enumerate(data):
            if "response" not in item:
                continue

            row_data = {"index": item.get("index", i)}  # Use item position as fallback
            response = item["response"]

            try:
                labels = self.extract_labels(response)
            except Exception as e:
                # Try to parse from raw_response if available
                if "raw_response" in item:
                    try:
                        response_string = item["raw_response"]
                        response_string_cleaned = response_string.replace(
                            "```json", ""
                        ).replace("```", "")
                        response = json.loads(response_string_cleaned)
                        labels = self.extract_labels(response)
                    except Exception:
                        # print(f"Error processing item {i} in {json_path}")
                        continue
                else:
                    # print(f"Error processing item {i} in {json_path}: {e}")
                    continue

            # Add labels to row data
            combined_reasoning_parts = []
            for label_dict in labels:
                category = label_dict["category"]
                row_data[category] = label_dict["label"]

                reasoning = label_dict["reasoning"]
                if reasoning:
                    combined_reasoning_parts.append(f"{category}: {reasoning}")

            row_data["combined_reasoning"] = "\n".join(combined_reasoning_parts)
            all_data.append(row_data)

        df = pd.DataFrame(all_data)

        # Ensure all expected columns are present
        if not df.empty:
            # Add missing label columns with default value 0
            for label in self.ALL_LABELS:
                if label not in df.columns:
                    df[label] = 0

            # Keep only relevant columns
            columns_to_keep = ["index"] + self.ALL_LABELS + ["combined_reasoning"]
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

        # Convert yes/True/y to 1 and no/NaN to 0 for label columns, keep other values as is

        for label in self.ALL_LABELS:
            if label in df.columns:
                df[label] = df[label].apply(self.convert_to_int)

        # First, find and convert any dictionaries in the dataframe
        # for label in self.ALL_LABELS:
        #     if label in df.columns:
        #         # Convert dictionaries to strings or another value
        #         df[label] = df[label].apply(
        #             lambda x: "dict_value" if isinstance(x, dict) else x
        #         )

        # Process each value individually without using .unique()
        all_label_values = set()
        for label in self.ALL_LABELS:
            if label in df.columns:
                for val in df[label]:
                    if not isinstance(val, dict):  # Skip dictionaries
                        all_label_values.add(val)

        # Define a type-aware sorting function
        def safe_sort_key(val):
            # Sort by type name first, then by value
            return (str(type(val)), str(val))

        # print("\nUnique values found across all label columns:")
        # # print(all_label_values)

        return df, all_label_values

    def process_directory(self, directory: str) -> Dict[str, pd.DataFrame]:
        """Process all JSON files in a directory.

        Args:
            directory: Directory containing JSON files

        Returns:
            Dictionary mapping file paths to processed DataFrames
        """
        results = {}

        total_label_values = set()
        for file in os.listdir(directory):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(directory, file)
            df, all_label_values = self.load_json(file_path)

            if not df.empty:
                results[file_path] = df
                print(f"Processed {file_path}: {df.shape} rows/columns")

                # add the values in the total_label_values set
                total_label_values.update(all_label_values)

        return results


if __name__ == "__main__":
    processor = LabelProcessor()
    results = processor.process_directory("open_models")

    # results = processor.load_json("open_models/ollama_labels_qwen2.5:latest.json")

    # # Print NaN counts for each column
    # print("\nNaN counts per column:")
    # print(results.isna().sum())

    # Print summary of results
    # for path, df in results.items():
    #     print(f"\nFile: {path}")
    #     print(f"Shape: {df.shape}")
    #     print(f"Columns: {df.columns.tolist()}")
