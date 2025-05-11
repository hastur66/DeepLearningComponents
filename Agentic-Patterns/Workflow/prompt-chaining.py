from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


client = genai.Client(api_key=GEMINI_API_KEY)

original_text = "Large language models are powerful AI systems trained on vast amounts of text data. They can generate human-like text, translate languages, write different kinds of creative content, and answer your questions in an informative way."
prompt_1 = f"Summarize the following text in one sentence: {original_text}"

response_1 = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt_1
)
summary = response_1.text.strip()
print(f"Summary: {summary}")

prompt_2 = f"Translate the following summary into French, only return the translation, no other text: {summary}"

response_2 = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt_2
)

translation = response_2.text.strip()
print(f"Translation: {translation}")
