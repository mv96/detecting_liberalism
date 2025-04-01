import os
from openai import OpenAI, File
import pandas as pd
import json

# If the OPENAI_API_KEY environment variable is not set,
# try to load the API key from key.txt and set the variable.
if os.getenv("OPENAI_API_KEY") is None:
    try:
        with open("key.txt", "r") as key_file:
            api_key = key_file.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
        print("=" * 20)
        print("API key loaded from key.txt.")
    except Exception as e:
        print("Error: Could not load API key from key.txt:", e)

# Create the OpenAI client using the API key from the environment.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Retrieve your API key from an environment variable for security reasons.

# Labelling task prompt from data_labelling.py.
labelling_prompt = """
You are a helpful assistant. You are given a text and your task is to label it with one or more of the following categories. I have provided you with the definitions and examples for each category. Follow the below instructions carefully to do the task. The input text is given between the triple backticks.

Instructions for the task:
1. Read the text carefully.
2. Read the definitions and examples for each category carefully.
3. Determine which of the categories apply to the text. If a category applies to the text return "yes" else return "no".
4. Along with the label, provide a short explanation for why you chose the label.
5. Return the output in a JSON format. An example output is given below.

### Categories and Definitions:

1. Partisan State (partisan_state_label): Introducing some bias to state institutions and policies by irrationally claiming that they always prefer a certain group.

2. Closed Society (closed_society_label): Unmotivated closure for cultural differences or multiculturalism. Irrationality in opposing rational knowledge, science, or debate.

3. Power Concentration (power_concentration_label): Texts implying the reduction of checks and balances to take more power.

4. Euroscepticism (euroscepticism_label): Opposing the EU in an irrational or unmotivated way.

5. Economic Protectionism (economic_label): Explicitly wanting to limit or criticize free trade or economic exchanges, including globalization, and emphasizing the priority of the national economy.

6. Complaints of Censorship or Mistreatment (censorship_label): Complaints of censorship, unfair treatment, or that the state or institutions like the media are partial.

7. Anti-Immigration (immigration_label): Immotivated or exaggerated complaints about non-dominant cultural or ethnic groups. Arguing for the predominance of the dominant cultural or ethnic group.

```
Given the following text: "{input_text}" can you provide the label and reasoning for each category?
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


def get_ai_answer(prompt_text, model_name="o3-mini"):
    """
    Get an answer from OpenAI's API for a given prompt using the Chat Completion API.

    Args:
        prompt_text (str): The prompt/question to pass to the AI.

    Returns:
        str: The response provided by the AI.
    """
    if "o3" in model_name:
        try:
            response = client.chat.completions.create(
                model=model_name,  # You can adjust the model as needed.
                messages=[{"role": "system", "content": prompt_text}],
                response_format={"type": "json_object"},
            )
            # Extract the answer from the response.
            answer = response.model_dump()["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            return f"Error: {e}"
    else:
        try:
            response = client.chat.completions.create(
                model=model_name,  # You can adjust the model as needed.
                messages=[{"role": "system", "content": prompt_text}],
                response_format={"type": "json_object"},
            )
            # Extract the answer from the response.
            answer = response.model_dump()["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            return f"Error: {e}"


def csv_to_jsonl(
    csv_path: str,
    jsonl_path: str,
    model_name: str,
    save_flag: bool = True,
    column_name: str = "New Example",
) -> list:
    """
    Convert all values from the CSV's "New Example" column into a list of JSON objects
    and optionally save them as a JSONL file. Each JSON object is structured as follows:

    {
      "custom_id": "request-<number>",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
         "model": <model_name>,
         "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": <generated prompt>}
          ],
         "max_tokens": 1000
      }
    }

    Args:
        csv_path (str): Path to the CSV file.
        jsonl_path (str): Path where the JSONL file should be saved.
        model_name (str): The model name to include in each JSON object.
        save_flag (bool, optional): If True, the JSONL file is saved to disk.

    Returns:
        list: A list of JSON objects for the JSONL content.
    """
    # Load the CSV file; assuming the column to convert is named "New Example"
    df = pd.read_csv(csv_path)
    examples = df[column_name]

    jsonl_lines = []
    for idx, example in enumerate(examples, start=1):
        # Base JSON structure
        json_line = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": labelling_prompt.format(input_text=example),
                    },
                ],
            },
        }

        # Add the appropriate tokens parameter based on model
        if "o3" in model_name:
            json_line["body"]["max_completion_tokens"] = 10000
        else:
            json_line["body"]["max_tokens"] = 1000

        jsonl_lines.append(json_line)

    if save_flag:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for line in jsonl_lines:
                f.write(json.dumps(line) + "\n")
        print(f"JSONL file saved to {jsonl_path}")
    else:
        print(
            "Save flag is false. JSONL file was not saved, but the JSONL object has been generated."
        )

    return jsonl_lines


def batch_predict(
    jsonl_file: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: dict = None,
) -> dict:
    """
    Execute a batch prediction job using the provided JSONL file with OpenAI's API.

    Args:
        jsonl_file (str): Path to the JSONL file containing the batch input requests.
        endpoint (str): API endpoint to be used for predictions (default: "/v1/chat/completions").
        completion_window (str): Allowed time period for the batch job to complete (default: "24h").
        metadata (dict): Optional metadata to attach to the batch job (default: {"description": "Batch prediction job"}).

    Returns:
        dict: The response from the batch job creation API call.
    """
    if metadata is None:
        metadata = {"description": "Batch prediction job"}

    try:
        # Upload the JSONL file as a batch input file.
        with open(jsonl_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        print("Batch input file successfully uploaded:", batch_input_file)

        # Create the batch prediction job using the uploaded file.
        response = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        print("Batch prediction job created:", response)
        return response

    except Exception as e:
        print("Error during batch prediction:", e)
        return {"error": str(e)}


def create_batch_job(batch_input_file, task_name):
    """
    Create a batch prediction job using the provided batch input file object.

    Args:
        batch_input_file: The uploaded file object returned from client.files.create.

    Returns:
        dict: The response from the batch job creation API call.
    """
    batch_input_file_id = batch_input_file.id
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": task_name},
    )
    return response


def display_batch_results(batch_id: str):
    """
    Fetch and display the batch information and its results for the given batch_id.

    Args:
        batch_id (str): The ID of the batch job.

    Returns:
        dict: The batch results if retrieval is successful, otherwise None.
    """
    try:
        # Retrieve the batch details.
        batch = client.batches.retrieve(batch_id)
        print("👇batch status:👇")
        print("Batch retrieved. Batch ID:", batch.id)
        print("Batch status:", batch.status)
        print("Batch results:", batch.request_counts)

    except Exception as e:
        print("Error retrieving batch or results:", e)
        return None


def display_file_results(file_id: str):
    """
    Retrieve and display the complete file content for the given file_id.
    """
    try:
        # Retrieve the file content.
        file_content = client.files.content(file_id)
        # Print the complete file object for inspection.
        return file_content
    except Exception as e:
        print("Error retrieving file content:", e)
        return None


def save_results_as_jsonl(
    file_content_obj, output_filename: str = "batch_results.jsonl"
):
    """
    Save the batch results into a JSONL file.

    This version handles various file object types, including HttpxBinaryResponseContent.
    """
    try:
        # If the object has a 'content' attribute (common for binary responses)
        if hasattr(file_content_obj, "content"):
            file_bytes = file_content_obj.content
            file_text = file_bytes.decode("utf-8")
        # If it's directly bytes, decode it
        elif isinstance(file_content_obj, bytes):
            file_text = file_content_obj.decode("utf-8")
        # As a last resort, convert to string
        else:
            file_text = str(file_content_obj)

        if not file_text.strip():
            print("No text content found in file_content.")
            return

        # Attempt to parse the text as JSON
        data = json.loads(file_text)
        with open(output_filename, "w", encoding="utf-8") as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item) + "\n")
            else:
                f.write(json.dumps(data) + "\n")
        print("Results saved as JSONL to", output_filename)
    except Exception as e:
        print("Error saving results as JSONL:", e)


def view_file_content(file_id: str):
    """
    Retrieve and print the decoded content from the given file_id.
    """
    try:
        file_content = client.files.content(file_id)
        # Explore attributes (optional)
        print("Attributes available:", dir(file_content))
        # print(file_content.content)

        response = client.files.retrieve(file_id)
        # print(response)

        # Assuming the response contains a 'content' attribute that is bytes:
        if hasattr(file_content, "content"):
            decoded_text = file_content.content.decode(
                "utf-8"
            )  # result = files.retrieve(result_file_id).decode("utf-8")
            print("Decoded file content:")
            # print(decoded_text)
        else:
            print("No content attribute found. Raw object:")
            print(file_content)
    except Exception as e:
        print("Error retrieving file content:", e)


if __name__ == "__main__":
    # Set batch_id to a non-empty string if you wish to skip job creation and directly fetch results.
    # If providing a batch_id manually, you must also supply the corresponding file id for file results.

    # Gpt-4o-mini call batch_67bc437d428881908eb5292de47d665d file-QFJjzvnaJB2mCbYM5uFzbq
    # o3-mini call batch_67bc78d20cfc8190ba95952be568f772 file-Nn6Pw7FVmprEXnNAU3Nijg
    batch_id = None
    file_id = None
    file_name = "batch_results.jsonl"
    # file-SW4CsQu5b7Mp43n1EpdTYo
    # batch_67b47a81625c8190a1b832bb12163d07

    # to check the status of the batch and fetch the results
    if batch_id is not None or file_id is not None:
        if batch_id is not None:
            print(f"Fetching results for provided Batch ID: {batch_id}")
            display_batch_results(batch_id)
        else:
            print("Please provide a batch id to fetch the results:")
            batch_id = input("Enter the batch ID: ").strip()
            display_batch_results(batch_id)

        # Prompt user for the file id corresponding to the batch's input file.
        if file_id is None:
            file_id = input(
                "Please enter the file ID associated with this batch: "
            ).strip()
            file_content = display_file_results(file_id)
            if file_content:
                print("File content:", file_content)
                save_results_as_jsonl(file_content)
        else:
            file_content = display_file_results(file_id)
            view_file_content(file_id)
            if file_content:
                # Print the raw file_content for inspection.
                # If file_content is a dict with a 'text' key, then use that.
                content_to_save = (
                    file_content.get("response", "")
                    if isinstance(file_content, dict)
                    else ""
                )
                if content_to_save:
                    save_results_as_jsonl(content_to_save, file_name)
                else:
                    print("No text content found in file_content.")
    else:
        # Define parameters for job creation.
        model_name = "o3-mini"
        csv_path = "701_new_examples.csv"
        jsonl_path = "output_requests.jsonl"
        task_name = "politics"  # custom name for the batch inference
        column_name = "translation"
        # Convert CSV examples to JSONL.
        jsonl_lines = csv_to_jsonl(
            csv_path, jsonl_path, model_name, column_name=column_name
        )

        # Upload the JSONL file as a batch input file.
        with open(jsonl_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        file_id = batch_input_file.id  # Capture the file ID for later use.
        print("👇Save the file id for retrieval purposes 👇")
        print(f"File ID:{file_id}")

        # Create the batch prediction job using the uploaded file.
        batch_job = create_batch_job(batch_input_file, task_name)
        batch_id = batch_job.id
        print("👇Save the batch id for tracking the progress of the batch 👇")
        print(f"Fetched Batch ID (use for tracking): {batch_id}")

        # Display batch details and results on the status of the batch
        display_batch_results(batch_id)

        # Use the correct file id when retrieving file content.
        # print("👇file results:👇")
        # file_content = display_file_results(file_id)
