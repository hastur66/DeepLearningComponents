import google.generativeai as genai
from google.generativeai import caching
import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ['API_KEY'])

path_to_video_file = 'Sherlock_Jr_FullMovie.mp4'

video_file = genai.upload_file(path=path_to_video_file)

while video_file.state.name == 'PROCESSING':
  print('Waiting for video to be processed.')
  time.sleep(2)
  video_file = genai.get_file(video_file.name)

print(f'Video processing complete: {video_file.uri}')

# Create a cache with a 5 minute TTL
cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-001',
    display_name='sherlock jr movie', # used to identify the cache
    system_instruction=(
        'You are an expert video analyzer, and your job is to answer '
        'the user\'s query based on the video file you have access to.'
    ),
    contents=[video_file],
    ttl=datetime.timedelta(minutes=5),
)


model = genai.GenerativeModel.from_cached_content(cached_content=cache)


response = model.generate_content([(
    'Introduce different characters in the movie by describing '
    'their personality, looks, and names. Also list the timestamps '
    'they were introduced for the first time.')])

print(response.usage_metadata)

print(response.text)
