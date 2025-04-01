import pandas as pd
import json
import os
import re
from json_repair import repair_json
import time
import asyncio
from ollama import AsyncClient
from tqdm.asyncio import tqdm as async_tqdm  # For async progress bars
from tqdm import tqdm  # Import regular tqdm for the overall progress bar


async def get_ollama_answer(
    prompt, model_name="llama2", temperature=0, ollama_client=None
):
    """
    Get an answer from Ollama's API for a given prompt using AsyncClient.
    """
    try:
        # If client is not provided, create a new one
        client = ollama_client or AsyncClient()

        response = await client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": temperature,
                "seed": 42,
            },
            stream=False,
        )
        return response["message"]["content"]
    except Exception as e:
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


async def request_json_format_fix(
    model_name, original_response, temperature=0, ollama_client=None
):
    """
    Ask the LLM to fix its non-JSON response by reformatting it into valid JSON.
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
"""

    # Request a fixed response
    fixed_response = await get_ollama_answer(
        fix_prompt, model_name, temperature, ollama_client
    )
    return fixed_response


async def process_single_example(idx, text, model_name, temperature, ollama_client):
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
    start_time = time.time()
    response = await get_ollama_answer(
        formatted_prompt,
        model_name,
        temperature,
        ollama_client,
    )
    processing_time = time.time() - start_time

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
        "processing_time_seconds": processing_time,
    }

    # Try to parse the response as JSON
    try:
        # Look for JSON array
        json_match = re.search(r"\[(.*?)\]", response, re.DOTALL)
        if json_match:
            json_str = "[" + json_match.group(1) + "]"
            try:
                result_entry["response"] = json.loads(json_str)
                result_entry["is_json"] = True
                return result_entry
            except json.JSONDecodeError:
                pass

        # Try JSON recovery
        recovered_json, success = attempt_json_recovery(response)
        if success:
            result_entry["response"] = recovered_json
            result_entry["is_json"] = True
            return result_entry

        # Request JSON format fix as a last resort
        retry_response = await request_json_format_fix(
            model_name, response, temperature, ollama_client
        )

        try:
            fixed_json = json.loads(retry_response)
            result_entry["response"] = fixed_json
            result_entry["is_json"] = True
            return result_entry
        except json.JSONDecodeError:
            recovered_json, success = attempt_json_recovery(retry_response)
            if success:
                result_entry["response"] = recovered_json
                result_entry["is_json"] = True
                return result_entry

        # If all parsing attempts fail
        result_entry["response"] = response
        result_entry["is_json"] = False

    except Exception as e:
        print(f"Error parsing response for row {idx}: {str(e)}")
        result_entry["response"] = response
        result_entry["is_json"] = False

    return result_entry


async def process_example(idx, text, model_name, temperature, client, server_url):
    """Process a single example without client-side concurrency control."""
    # print(f"Server {server_url} starting to process example {idx}")
    start_time = time.time()

    # Direct processing without semaphore
    result = await process_single_example(idx, text, model_name, temperature, client)
    if result is not None:
        # Add server and timing information to the result
        processing_time = time.time() - start_time
        result["server_info"] = {
            "server_url": server_url,
            "processing_time_seconds": processing_time,
        }
        # print(
        #     f"Server {server_url} completed example {idx} in {processing_time:.2f} seconds"
        # )
    else:
        print(f"Server {server_url} failed to process example {idx}")
    return result


async def process_csv_with_ollama(
    csv_path,
    model_name="llama2",
    temperature=0,
    column_name="New Example",
    output_path=None,
    server_endpoints=None,
    samples_per_batch=16,  # Can be int or list
):
    """
    Process examples with true parallel processing across multiple servers.
    Process in batches, distributed across servers based on samples_per_batch.

    Args:
        csv_path: Path to the CSV file
        model_name: Ollama model to use
        temperature: Temperature setting for generation
        column_name: CSV column to process
        output_path: Where to save results
        server_endpoints: List of Ollama server URLs
        samples_per_batch: Either:
            - An integer for total samples (distributed round-robin)
            - A list of integers specifying allocation per server [16,16] or [20,12]
    """
    start_time = time.time()

    if server_endpoints is None:
        server_endpoints = ["http://localhost:11434"]

    # Create a client for each server
    ollama_clients = [AsyncClient(host=endpoint) for endpoint in server_endpoints]
    num_servers = len(ollama_clients)

    # Handle samples_per_batch - can be int or list
    is_custom_allocation = isinstance(samples_per_batch, (list, tuple))

    if is_custom_allocation:
        # Validate server allocation
        if len(samples_per_batch) != num_servers:
            raise ValueError(
                f"When samples_per_batch is a list, its length ({len(samples_per_batch)}) "
                f"must match number of servers ({num_servers})"
            )
        # Calculate total batch size from the allocation
        batch_size = sum(samples_per_batch)
        print(f"Using custom server allocation: {samples_per_batch}")
    else:
        # Using simple integer for total batch size (round-robin distribution)
        batch_size = samples_per_batch

    print(f"Using {num_servers} Ollama servers with unlimited concurrent requests")
    print(f"Processing in batches of {batch_size} total samples")

    # Track which servers have been used
    servers_used = {endpoint: False for endpoint in server_endpoints}
    server_request_counts = {endpoint: 0 for endpoint in server_endpoints}
    server_total_processing_time = {endpoint: 0.0 for endpoint in server_endpoints}

    # Load existing results synchronously
    existing_ids = set()
    existing_results = []
    new_results = []  # Track new results separately
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                existing_ids = {
                    result["index"] for result in existing_results if "index" in result
                }
                print(f"Loaded {len(existing_results)} existing results")
            except json.JSONDecodeError:
                print("Warning: Could not parse existing results file. Starting fresh.")
                existing_results = []

    # Initialize the results list by combining existing and new results
    results = existing_results.copy()

    # Load CSV synchronously
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    print(f"Available columns: {df.columns.tolist()}")

    # Find rows to process (skip already processed) - synchronous operation
    rows_to_process = [
        (idx, row[column_name]) for idx, row in df.iterrows() if idx not in existing_ids
    ]
    total_remaining = len(rows_to_process)

    if total_remaining == 0:
        print("No new rows to process")
        return existing_results + new_results

    print(f"Processing {total_remaining} remaining examples")

    # Process examples in batches
    batch_size = (
        batch_size  # This is correct now, using the batch_size from the parameter
    )

    # Calculate number of batches
    num_batches = (total_remaining + batch_size - 1) // batch_size  # Ceiling division
    print(f"Will process in {num_batches} batches of {batch_size} samples")

    # Create an overall progress bar for all samples
    overall_pbar = tqdm(
        total=total_remaining,
        desc="Overall progress",
        position=0,
        unit="sample",
        smoothing=0.1,
    )

    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_remaining)
        current_batch = rows_to_process[start_idx:end_idx]

        print(
            f"\nStarting batch {batch_idx+1}/{num_batches} with {len(current_batch)} examples"
        )

        # Reset batch-specific tracking for each server
        batch_server_counts = {endpoint: 0 for endpoint in server_endpoints}
        batch_server_times = {endpoint: 0.0 for endpoint in server_endpoints}
        actual_server_processing_times = {
            endpoint: 0.0 for endpoint in server_endpoints
        }

        # Create tasks for this batch, distributing according to allocation or round-robin
        batch_tasks = []

        if is_custom_allocation:
            # Create server assignments based on allocation list
            server_assignments = []
            for server_idx, allocation in enumerate(samples_per_batch):
                server_assignments.extend([server_idx] * allocation)

            # Trim to match current_batch size if needed
            server_assignments = server_assignments[: len(current_batch)]

            # Assign tasks based on allocation
            for idx, ((example_idx, text), server_idx) in enumerate(
                zip(current_batch, server_assignments)
            ):
                server_url = server_endpoints[server_idx]
                client = ollama_clients[server_idx]

                # Comment out individual request prints
                # print(f"Assigning example {example_idx} to server {server_url}")

                batch_tasks.append(
                    asyncio.create_task(
                        process_example(
                            example_idx,
                            text,
                            model_name,
                            temperature,
                            client,
                            server_url,
                        )
                    )
                )
        else:
            # Original round-robin assignment
            server_idx = 0
            for idx, text in current_batch:
                # Choose next server in round-robin fashion
                server_url = server_endpoints[server_idx]
                client = ollama_clients[server_idx]

                # Comment out individual request prints
                # print(f"Assigning example {idx} to server {server_url}")

                batch_tasks.append(
                    asyncio.create_task(
                        process_example(
                            idx,
                            text,
                            model_name,
                            temperature,
                            client,
                            server_url,
                        )
                    )
                )

                # Move to next server
                server_idx = (server_idx + 1) % num_servers

        # Process this batch
        batch_start_time = time.time()
        batch_results = []  # Track results for this batch

        # Use position=1 for the batch progress bar to appear below the overall progress bar
        with async_tqdm(
            total=len(batch_tasks),
            desc=f"Batch {batch_idx+1}/{num_batches}",
            position=1,
            leave=False,
        ) as batch_pbar:
            for completed_task in asyncio.as_completed(batch_tasks):
                result = await completed_task
                if result is not None:
                    # Extract server information
                    if "server_info" in result:
                        server_url = result["server_info"]["server_url"]
                        processing_time = result["server_info"][
                            "processing_time_seconds"
                        ]

                        # Update global tracking
                        servers_used[server_url] = True
                        server_request_counts[server_url] += 1
                        server_total_processing_time[server_url] += processing_time

                        # Update batch tracking
                        batch_server_counts[server_url] += 1
                        batch_server_times[server_url] += processing_time
                        actual_server_processing_times[server_url] += processing_time

                    results.append(result)
                    new_results.append(result)  # Also track separately for clarity
                    batch_results.append(result)  # Add to this batch's results

                # Update both progress bars
                batch_pbar.update(1)
                overall_pbar.update(1)

                # Now this will work because start_time is defined
                elapsed = time.time() - start_time
                samples_done = overall_pbar.n
                if samples_done > 0:
                    samples_per_second = samples_done / elapsed
                    remaining_samples = total_remaining - samples_done
                    eta_seconds = (
                        remaining_samples / samples_per_second
                        if samples_per_second > 0
                        else 0
                    )
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    overall_pbar.set_description(f"Overall progress (ETA: {eta_str})")

        # Batch complete - save all results after receiving all responses
        if batch_results:  # Only save if we have results
            # Sort all results by index before saving
            results.sort(key=lambda x: x["index"])

            print(
                f"\nSaving {len(results)} total results in index order after batch completion"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        # Batch complete - show statistics
        batch_elapsed = time.time() - batch_start_time
        total_server_processing_time = sum(actual_server_processing_times.values())

        print(
            f"\n\nBatch {batch_idx+1}/{num_batches} completed in {batch_elapsed:.2f} seconds"
        )
        print(f"Batch server performance:")

        # Display individual server processing times with clear server numbering
        for i, (server, count) in enumerate(batch_server_counts.items(), 1):
            server_time = actual_server_processing_times[server]
            if count > 0:
                avg_time = server_time / count
                print(
                    f"  Server {i} ({server}): {count} samples, total time {server_time:.2f}s, avg {avg_time:.2f}s per sample"
                )
            else:
                print(f"  Server {i} ({server}): No samples processed in this batch")

        # Display total processing time across all servers
        print(f"  Combined server processing time: {total_server_processing_time:.2f}s")
        print(f"  Wall-clock batch time: {batch_elapsed:.2f}s")
        print(
            f"  Parallelization speedup: {(total_server_processing_time/batch_elapsed if batch_elapsed > 0 else 0):.2f}x"
        )

    # Close the overall progress bar when done
    overall_pbar.close()

    # Processing complete - show final statistics
    print("\n\nAll batches complete. Final server performance statistics:")
    for server, count in server_request_counts.items():
        if count > 0:
            avg_time = server_total_processing_time[server] / count
            print(
                f"  {server}: {count} total requests, avg time {avg_time:.2f} seconds"
            )

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


async def main():
    """Main async function to run the program."""
    start_time = time.time()

    # Define model and parameters
    model_name = "qwen2.5:0.5b"
    CSV_PATH = "701_new_examples.csv"
    COLUMN_NAME = "translation"
    OUTPUT_PATH = f"ollama_labels_{model_name}.json"

    print("Processing with parallel requests")

    # Labelling task prompt
    global labelling_prompt
    labelling_prompt = """
    You are a helpful assistant. You are given a text and your task is to label it
    with one or more of the following categories.

    Instructions for the task:
    1. Read the text carefully.
    2. Read the definitions for each category carefully.
    3. Determine which categories apply to the text. Return "yes" or "no" for each.
    4. Provide a short explanation for your label choice.
    5. Return the output in JSON format as shown in the example.

    ### Categories and Definitions:

    1. Partisan State (partisan_state_label): Introducing bias to state institutions by irrationally 
       claiming they always prefer a certain group.

    2. Closed Society (closed_society_label): Unmotivated closure for cultural differences or 
       multiculturalism. Irrationality in opposing rational knowledge, science, or debate.

    3. Power Concentration (power_concentration_label): Texts implying the reduction of 
       checks and balances to take more power.

    4. Euroscepticism (euroscepticism_label): Opposing the EU in an irrational or unmotivated way.

    5. Economic Protectionism (economic_label): Explicitly wanting to limit free trade or economic 
       exchanges, emphasizing priority of the national economy.

    6. Complaints of Censorship (censorship_label): Complaints of censorship, unfair treatment, 
       or that the state or media are partial.

    7. Anti-Immigration (immigration_label): Unmotivated complaints about non-dominant cultural 
       or ethnic groups. Arguing for predominance of the dominant cultural/ethnic group.

    ```
    Given the following text: "{input_text}" generate the output in the following JSON format:
    ```
    Example output:
    [
        {{"category": "partisan_state_label", "label": "yes", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "closed_society_label", "label": "no", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "power_concentration_label", "label": "no", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "euroscepticism_label", "label": "yes", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "economic_label", "label": "no", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "censorship_label", "label": "yes", "reasoning": "reasoning in at most 25 words"}},
        {{"category": "immigration_label", "label": "yes", "reasoning": "reasoning in at most 25 words"}}
    ]
    """

    # Set the server endpoints
    server_endpoints = ["http://localhost:11434", "http://localhost:11435"]

    # Example 1: Even distribution (16 samples per server)
    # samples_per_batch = 32  # Distributed evenly (round-robin)

    # Example 2: Custom allocation (server 1: 20 samples, server 2: 12 samples)
    samples_per_batch = [16, 16]  # Custom allocation per server

    results = await process_csv_with_ollama(
        csv_path=CSV_PATH,
        model_name=model_name,
        temperature=0,
        column_name=COLUMN_NAME,
        output_path=OUTPUT_PATH,
        server_endpoints=server_endpoints,
        samples_per_batch=samples_per_batch,  # Can be int or list
    )

    print(f"Processed {len(results)} examples")
    json_success = sum(1 for r in results if r.get("is_json", False))
    json_failed = len(results) - json_success
    print(f"Summary: {json_success} responses successfully parsed as JSON")
    print(f"         {json_failed} responses stored as raw text (JSON parsing failed)")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
