from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = AsyncAnthropic(api_key=API_KEY)

async def main():
    response = await anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT}How many moons does Jupiter have?{AI_PROMPT}",
    )
    print(response.completion)
	
asyncio.run(main())
