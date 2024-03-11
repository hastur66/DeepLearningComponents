from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = AsyncAnthropic(api_key=API_KEY)

async def main():
    stream = await anthropic.completions.create(
        prompt=f"{HUMAN_PROMPT}Please write a blog post on Neural Networks and use Markdown format. Kindly make sure that you proofread your work for any spelling, grammar, or punctuation errors.{AI_PROMPT}",
    	max_tokens_to_sample=350,
    	model="claude-2.1",
    	stream=True,
    )
    async for completion in stream:
    	print(completion.completion, end="", flush=True)
	
asyncio.run(main())
