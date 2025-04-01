import asyncio
from ollama import AsyncClient
import time
from tqdm.asyncio import tqdm

prompt = "Why is the sky blue?"
count = 128
prompt_list = [prompt for _ in range(count)]
prompt_list = [prompt_list[i] for i in range(count)]


async def chat(prompt_text):
    message = {"role": "user", "content": prompt_text}
    response = await AsyncClient().chat(
        model="qwen2.5:0.5b",
        messages=[message],
        options={"temperature": 0},  # Set temperature to 0
    )
    return response


async def main():
    # Create a list of chat tasks for all prompts
    tasks = [chat(prompt_text) for prompt_text in prompt_list]
    # Run all tasks concurrently with progress bar and get results
    results = await tqdm.gather(*tasks, desc="Processing requests")
    return results


# Run the main async function and measure time
start_time = time.time()
results = asyncio.run(main())
end_time = time.time()
elapsed_time = end_time - start_time
print(len(results))

print(f"\nProcessed {len(results)} responses")
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Average time per request: {elapsed_time/len(results):.2f} seconds")
# Optionally print the first result to check
print("First response:", results[0])
