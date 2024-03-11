from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = Anthropic(
	api_key=API_KEY,
	)

response = anthropic.messages.create(
	model="claude-3-opus-20240229", # claude-3-sonnet-20240229
	max_tokens=350,
	messages=[
        {"role": "user", "content": "Hello, there"}],
	)
	
print(f"Response: {response}")
print(f"Completion: {response.content}")
