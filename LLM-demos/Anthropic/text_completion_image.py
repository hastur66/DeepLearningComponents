from anthropic import Anthropic
import base64
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

anthropic = Anthropic(
	api_key=API_KEY,
	)
	
image1_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image1_media_type = "image/jpeg"
image1_data = base64.b64encode(httpx.get(image1_url).content).decode("utf-8")


response = anthropic.messages.create(
	model="claude-3-opus-20240229", # claude-3-sonnet-20240229
	max_tokens=350,
	messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ],
        }
        ],
	)
	
print(f"Response: {response}")
