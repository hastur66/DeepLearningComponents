from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = Anthropic(
	api_key=API_KEY,
	)

response = anthropic.messages.create(
	model="claude-2.1",
	system="Respond in shakespearean english", # To use system prompts with the Messages API, set the system parameter
	max_tokens=350,
	 messages=[
        {"role": "user", "content": "How do I learn Python in a week?"} ], # <-- user prompt
	)
	
print(f"Response message: {response.content}")
