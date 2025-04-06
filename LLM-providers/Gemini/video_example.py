import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

video_file_name = "BigBuckBunny_320x180.mp4"

print(f"Uploading file...")

video_file = genai.upload_file(path=video_file_name)

print(f"Completed upload: {video_file.uri}")

# check file processing status
while video_file.state.name == "PROCESSING":
    print('Waiting for video to be processed.')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)
print(f'Video processing complete: ' + video_file.uri)


prompt = "Describe this video."

model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

print("Making LLM inference request...")

response = model.generate_content([prompt, video_file],
                                  request_options={"timeout": 600})
print(response.text)

# optional automatically deleted after 2 days or manual delete
# genai.delete_file(video_file.name)
# print(f'Deleted file {video_file.uri}')