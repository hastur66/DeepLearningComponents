import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

audio_file_name = "sample.mp3"

print(f"Uploading file...")

audio_file = genai.upload_file(path=audio_file_name)

print(f"Completed upload: {audio_file.uri}")


prompt = "Listen carefully to the following audio file. Provide a brief summary."

model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

print("Making LLM inference request...")

response = model.generate_content([prompt, audio_file])
print(response.text)

print(model.count_tokens([audio_file]))