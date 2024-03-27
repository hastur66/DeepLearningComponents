from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = Anthropic(
	api_key=API_KEY,
	)

response = anthropic.completions.create(
	model="claude-2.1",
	max_tokens_to_sample=350,
	prompt=f"{HUMAN_PROMPT} How do I learn Python in a week?{AI_PROMPT}",
	)
	
print(f"Response: {response}")
print(f"Completion: {response.completion}")
