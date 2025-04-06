import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)


input = [
    "The dog is barking",
    "The cat is purring",
    "The bear is growling"
]


response = genai.embed_content(
    model="models/embedding-001",
    content=input,
    task_type="retrieval_query",
)["embedding"]


print(response)
print(np.asarray(response).shape)
