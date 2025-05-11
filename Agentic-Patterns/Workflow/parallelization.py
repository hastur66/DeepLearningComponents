from google import genai
from dotenv import load_dotenv
import asyncio
import time
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_content(prompt: str) -> str:
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()

async def parallel_tasks():
    topic = "a friendly robot exploring a jungle"
    prompts = [
        f"Write a short, adventurous story idea about {topic}.",
        f"Write a short, funny story idea about {topic}.",
        f"Write a short, mysterious story idea about {topic}."
    ]
    start_time = time.time()

    tasks = [generate_content(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    print(results)

    end_time = time.time()
 
    print(f"Time taken: {end_time - start_time} seconds")
 
    print("\n--- Individual Results ---")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}\n")
 
    story_ideas = '\n'.join([f"Idea {i+1}: {result}" for i, result in enumerate(results)])
    aggregation_prompt = f"Combine the following three story ideas into a single, cohesive summary paragraph:{story_ideas}"
    aggregation_response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=aggregation_prompt
    )
    return aggregation_response.text

result = asyncio.run(parallel_tasks())
print(f"\n--- Aggregated Summary ---\n{result}")
