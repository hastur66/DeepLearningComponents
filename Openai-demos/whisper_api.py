import openai
from dotenv import load_dotenv
import os
from pydub import AudioSegment

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

filename = 'data/2830-3980-0043.wav'

# audio_file= open(filename, "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)
# print(transcript)
# audio_file.close()

def speech_to_text(filename):
    """
    Transcribe audio file into text
    """
    
    try:
        # sound = AudioSegment.from_file(filename)
        # sound.export(filename, format="mp3")
        with open(filename, 'rb') as audio_file:
            transcript = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
        )
        return transcript
    except Exception as e:
        print(e)
        transcript = "Error"

        return transcript
        

print(speech_to_text(filename))