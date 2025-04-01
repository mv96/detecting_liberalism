import pandas as pd
import json
import requests
import sys
import os
from tqdm import tqdm
import re
from json_repair import repair_json
import time

# Create a single global session for all requests
session = requests.Session()


def list_ollama_models():
    """
    List all available models in Ollama.

    Returns:
        List of model names or None if an error occurs
    """
    url = "http://localhost:11434/api/tags"

    try:
        response = session.get(url)
        response.raise_for_status()
        result = response.json()
        models = [model["name"] for model in result.get("models", [])]
        return models
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving models: {e}")
        print("Make sure Ollama is running.")
        return None


def get_ollama_answer(prompt, model_name="llama2", temperature=0):
    """
    Get an answer from Ollama's API for a given prompt.

    Args:
        prompt (str): The fully formatted prompt to send to the model.
        model_name (str): The Ollama model to use.
        temperature (float): Controls randomness (0.0-1.0).

    Returns:
        str: The response provided by the model.
    """
    # Record start time
    start_time = time.time()

    # Prepare the request to Ollama API
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "seed": 42,
    }

    try:
        response = session.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and the model is installed.")
        return f"Error: {e}"


def attempt_json_recovery(text):
    """
    Try to recover JSON from malformed responses using jsonrepair library
    and other fallback techniques.

    Args:
        text (str): The text containing potential JSON

    Returns:
        tuple: (recovered_json, success_bool)
    """
    # First try using jsonrepair library
    try:
        repaired_json = repair_json(text)
        parsed_json = json.loads(repaired_json)
        return parsed_json, True
    except Exception:
        pass

    # Fallback to our original techniques if jsonrepair fails

    # 1. Remove any text before the first { and after the last }
    try:
        start_idx = text.find("{")
        if start_idx >= 0:
            end_idx = text.rfind("}")
            if end_idx > start_idx:
                cuttext = text[start_idx : end_idx + 1]
                # Try to parse it
                return json.loads(cuttext), True
    except json.JSONDecodeError:
        pass

    # 2. Look for JSON array
    try:
        start_idx = text.find("[")
        if start_idx >= 0:
            end_idx = text.rfind("]")
            if end_idx > start_idx:
                cuttext = text[start_idx : end_idx + 1]
                # Try to parse it
                return json.loads(cuttext), True
    except json.JSONDecodeError:
        pass

    # 3. Common JSON formatting issues
    # Replace single quotes with double quotes
    try:
        fixed_text = text.replace("'", '"')
        return json.loads(fixed_text), True
    except json.JSONDecodeError:
        pass

    # 4. Extract individual JSON objects
    try:
        objects = re.findall(r"{.*?}", text, re.DOTALL)
        if objects:
            valid_objects = []
            for obj in objects:
                try:
                    valid_objects.append(json.loads(obj))
                except json.JSONDecodeError:
                    continue
            if valid_objects:
                return valid_objects, True
    except Exception:
        pass

    return text, False


def request_json_format_fix(model_name, original_response, temperature=0):
    """
    Ask the LLM to fix its non-JSON response by reformatting it into valid JSON.

    Args:
        model_name (str): The Ollama model to use.
        original_response (str): The original malformed response from the LLM.
        temperature (float): Temperature setting.

    Returns:
        str: Reformatted response that should be valid JSON.
    """
    # Create a prompt asking the model to fix its previous output
    fix_prompt = f"""Your previous response was not in valid JSON format. 
Please reformat your answer as a valid JSON array of objects, where each object has 
"category", "label", and "reasoning" fields.

Your previous response was:
```
{original_response}
```

Please output ONLY valid JSON following this structure:
[
  {{"category": "category_name", "label": "yes/no", "reasoning": "brief explanation"}}
]

Ensure you use double quotes for strings and proper JSON syntax.
"""

    # Request a fixed response
    fixed_response = get_ollama_answer(fix_prompt, model_name, temperature)
    return fixed_response


def process_single_example(idx, text, model_name, temperature):
    """
    Process a single example with Ollama.

    Args:
        idx (int): Index of the example.
        text (str): Text to analyze.
        model_name (str): Name of the Ollama model to use.
        temperature (float): Temperature setting.

    Returns:
        dict: Result dictionary or None if an error occurs.
    """
    if pd.isna(text) or text == "":
        print(f"Warning: Empty text in row {idx}, skipping")
        return None

    # Format the prompt
    formatted_prompt = labelling_prompt.format(input_text=text)

    # Process with Ollama
    response = get_ollama_answer(formatted_prompt, model_name, temperature)

    # Check for errors in the response
    if "Error" in response:
        print(f"Error processing row {idx}: {response}")
        return None  # Return None if there's an error

    # Store the result
    result_entry = {
        "index": int(idx),
        "input_text": text,
        "model": model_name,
        "prompt_text": formatted_prompt,  # Add the prompt text to the result
        "raw_response": response,  # Always store the raw response for debugging
    }

    # Try to parse the response as JSON
    try:
        # First try to find a complete JSON array
        json_match = re.search(r"\[(.*?)\]", response, re.DOTALL)
        if json_match:
            json_str = "[" + json_match.group(1) + "]"
            try:
                parsed_response = json.loads(json_str)
                result_entry["response"] = parsed_response
                result_entry["is_json"] = True
                return result_entry
            except json.JSONDecodeError:
                # If the found JSON is malformed, we'll try the more robust approach below
                pass

        # If no valid JSON array was found, try to extract individual JSON objects
        # This handles cases where the model outputs partial or streaming responses
        json_objects = []

        # Look for category objects in the format {"category": "...", "label": "...", "reasoning": "..."}
        category_matches = re.finditer(
            r'{\s*"category":\s*"([^"]+)",'
            + r'\s*"label":\s*"([^"]+)",'
            + r'\s*"reasoning":\s*"([^"]+)"\s*}',
            response,
            re.DOTALL,
        )

        for match in category_matches:
            category = match.group(1)
            label = match.group(2)
            reasoning = match.group(3)

            # Clean up reasoning text (remove extra whitespace, line breaks)
            reasoning = re.sub(r"\s+", " ", reasoning).strip()

            json_objects.append(
                {"category": category, "label": label, "reasoning": reasoning}
            )

        if json_objects:
            result_entry["response"] = json_objects
            result_entry["is_json"] = True
            return result_entry

        # If we still don't have valid JSON, try once more with a more general approach
        try:
            # Try to extract any JSON-like structures
            all_json_like = re.findall(r'{(?:[^{}]|"(?:\\.|[^"\\])*")*}', response)
            if all_json_like:
                combined = "[" + ",".join(all_json_like) + "]"
                parsed_response = json.loads(combined)
                result_entry["response"] = parsed_response
                result_entry["is_json"] = True
                return result_entry
        except json.JSONDecodeError:
            pass

        # As a last resort, try to parse the entire response as JSON
        try:
            parsed_response = json.loads(response)
            result_entry["response"] = parsed_response
            result_entry["is_json"] = True
        except json.JSONDecodeError:
            # Try our JSON repair as a last resort
            recovered_json, success = attempt_json_recovery(response)
            if success:
                result_entry["response"] = recovered_json
                result_entry["is_json"] = True
                result_entry["json_recovered"] = True
                print(f"Successfully recovered JSON for row {idx} using repair tools")
                return result_entry

            # If JSON recovery failed, show the full response
            print(f"Error parsing JSON for row {idx}")
            print(f"Raw response: {response[:150]}...")  # Show only first 150 chars

            # If not valid JSON, store raw response
            result_entry["response"] = response
            result_entry["is_json"] = False

            # Save problematic responses to a separate debug file
            debug_file = "json_parse_failures.txt"
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- EXAMPLE {idx} ---\n")
                f.write(f"PROMPT:\n{formatted_prompt}\n\n")
                f.write(f"RAW RESPONSE:\n{response}\n")

        # NEW: As a final fallback, ask the model to fix its JSON format
        print(f"Attempting to request JSON format fix for row {idx}")
        retry_response = request_json_format_fix(model_name, response, temperature)

        # Try to parse the reformatted response
        try:
            # First try direct JSON parsing
            fixed_json = json.loads(retry_response)
            result_entry["response"] = fixed_json
            result_entry["is_json"] = True
            result_entry["json_reformatted"] = True
            print(f"Successfully reformatted JSON for row {idx}")
            return result_entry
        except json.JSONDecodeError:
            # If direct parsing fails, try our recovery techniques on the retry
            recovered_json, success = attempt_json_recovery(retry_response)
            if success:
                result_entry["response"] = recovered_json
                result_entry["is_json"] = True
                result_entry["json_recovered"] = True
                result_entry["json_reformatted"] = True
                print(f"Successfully recovered reformatted JSON for row {idx}")
                return result_entry

        # If reformatting fails too, show the full response instead of truncating it
        print(f"Error parsing JSON for row {idx}, even after reformatting attempt")
        print(f"Raw response: {response[:150]}...")  # Show first 150 chars
        # If not valid JSON, store raw response
        result_entry["response"] = response
        result_entry["is_json"] = False

        # Save problematic responses to a separate debug file
        debug_file = "json_parse_failures.txt"
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- EXAMPLE {idx} ---\n")
            f.write(f"PROMPT:\n{formatted_prompt}\n\n")
            f.write(f"ORIGINAL RESPONSE:\n{response}\n\n")
            f.write(f"RETRY RESPONSE:\n{retry_response}\n")

    except Exception as e:
        print(f"Unexpected error parsing response for row {idx}: {str(e)}")
        print(f"Raw response: {response}")
        # Store raw response
        result_entry["response"] = response
        result_entry["is_json"] = False

        # Save error cases to debug file
        debug_file = "json_parse_failures.txt"
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- EXAMPLE {idx} (ERROR) ---\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"PROMPT:\n{formatted_prompt}\n\n")
            f.write(f"RAW RESPONSE:\n{response}\n")

    return result_entry


def process_csv_with_ollama(
    csv_path,
    model_name="llama2",
    temperature=0,
    column_name="New Example",
    output_path=None,
):
    """
    Process examples in a CSV file using Ollama.

    Args:
        csv_path (str): Path to the CSV file.
        model_name (str): Name of the Ollama model to use.
        temperature (float): Temperature setting (0.0-1.0).
        column_name (str): Name of the column containing text to analyze.
        output_path (str): Path to save results.

    Returns:
        list: List of results.
    """
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_ollama_results.json"

    # Load existing results to skip already processed IDs
    existing_ids = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            existing_ids = {
                result["index"] for result in existing_results if "index" in result
            }
            results.extend(existing_results)

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    print(f"Available columns: {df.columns.tolist()}")

    # Process examples sequentially with enhanced progress bar
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing {model_name}",
        unit="examples",
        ncols=100,  # Width of the progress bar
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        if idx in existing_ids:
            print(f"Skipping already processed example {idx}")
            continue
        text = row[column_name]
        result = process_single_example(idx, text, model_name, temperature)
        if result:
            if result.get("is_json", False) or "Error" not in result.get(
                "raw_response", ""
            ):
                # Append result to results
                results.append(result)
                # Save results incrementally
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    # ============= CONFIGURATION SETTINGS =============
    # Set your parameters here
    start_time = time.time()

    # get max workers for ollama model
    model_names = ["qwen2.5:0.5b"]
    for model_name in model_names:
        CSV_PATH = "701_new_examples.csv"
        COLUMN_NAME = "translation"
        MODEL_NAME = model_name
        TEMPERATURE = 0
        OUTPUT_PATH = f"ollama_labels_{MODEL_NAME}.json"

        # Labelling task prompt from data_labelling.py.
        labelling_prompt = """
        You are a helpful assistant. You are given a text and your task is to label it
        with one or more of the following categories. I have provided you with the
        definitions and examples for each category. Follow the below instructions
        carefully to do the task. The input text is given between the triple backticks.

        Instructions for the task:
        1. Read the text carefully.
        2. Read the definitions and examples for each category carefully.
        3. Determine which of the categories apply to the text. If a category applies
        to the text return "yes" else return "no".
        4. Along with the label, provide a short explanation for why you chose the label.
        5. Return the output in a JSON format. An example output is given below.

        ### Categories and Definitions:

        1. Partisan State (partisan_state_label): Introducing some bias to state
        institutions and policies by irrationally claiming that they always prefer
        a certain group.

        2. Closed Society (closed_society_label): Unmotivated closure for cultural
        differences or multiculturalism. Irrationality in opposing rational knowledge,
        science, or debate.

        3. Power Concentration (power_concentration_label): Texts implying the reduction
        of checks and balances to take more power.

        4. Euroscepticism (euroscepticism_label): Opposing the EU in an irrational or
        unmotivated way.

        5. Economic Protectionism (economic_label): Explicitly wanting to limit or
        criticize free trade or economic exchanges, including globalization, and
        emphasizing the priority of the national economy.

        6. Complaints of Censorship or Mistreatment (censorship_label): Complaints of
        censorship, unfair treatment, or that the state or institutions like the
        media are partial.

        7. Anti-Immigration (immigration_label): Immotivated or exaggerated complaints
        about non-dominant cultural or ethnic groups. Arguing for the predominance
        of the dominant cultural or ethnic group.

        ```
        Given the following text: "{input_text}" generate the output in the following JSON format:
        ```
        Example output:
        [
            {{
                "category": "partisan_state_label",
                "label": "yes",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "closed_society_label",
                "label": "no",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "power_concentration_label",
                "label": "no",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "euroscepticism_label",
                "label": "yes",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "economic_label",
                "label": "no",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "censorship_label",
                "label": "yes",
                "reasoning": "reasoning in atmost 25 words"
            }},
            {{
                "category": "immigration_label",
                "label": "yes",
                "reasoning": "reasoning in atmost 25 words"
            }}
        ]
        """

        try:
            print("Checking for available Ollama models...")
            models = list_ollama_models()
            if not models:
                print("Could not connect to Ollama. Make sure it's running.")
                sys.exit(1)

            results = process_csv_with_ollama(
                csv_path=CSV_PATH,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                column_name=COLUMN_NAME,
                output_path=OUTPUT_PATH,
            )

            print(f"Processed {len(results)} examples")
            json_success = sum(1 for r in results if r.get("is_json", False))
            json_failed = len(results) - json_success
            print(f"Summary: {json_success} responses successfully parsed as JSON")
            print(
                f"         {json_failed} responses stored as raw text (JSON parsing failed)"
            )
            if json_failed > 0:
                print(f"Problematic responses saved to json_parse_failures.txt")

        finally:
            session.close()

        elapsed = time.time() - start_time
        print(f"Total processing time: {elapsed:.2f} seconds")
