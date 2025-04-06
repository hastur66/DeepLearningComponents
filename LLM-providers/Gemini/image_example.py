import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro-vision')

img = PIL.Image.open('data/image.jpg')

response = model.generate_content(img)

print(response.text)

response_descriptive = model.generate_content(["""Write a short, engaging blog post based on this picture.
                                               It should include a description of the meal in the photo and talk about
                                               my journey meal prepping.""", 
                                               img],
                                               stream=True)

response_descriptive.resolve()
print("Descriptive response: ", response_descriptive.text)