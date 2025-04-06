import google.generativeai as palm
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PALM_API_KEY")

# Set the API key
palm.configure(api_key=api_key)

response = palm.chat(messages=["Hello ."])

print(response.last)

response.reply("Can you tell me a joke?")