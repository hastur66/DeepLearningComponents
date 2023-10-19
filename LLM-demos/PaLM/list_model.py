import google.generativeai as palm
import pprint
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PALM_API_KEY")

# Set the API key
palm.configure(api_key=api_key)

for model in palm.list_models():
    pprint.pprint(model)
    