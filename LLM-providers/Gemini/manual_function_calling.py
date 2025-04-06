import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.protobuf.struct_pb2 import Struct
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)


def find_movies(description: str, location: str = ""):
    """find movie titles currently playing in theaters based on any description, genre, title words, etc.

    Args:
        description: Any kind of description including category or genre, title words, attributes, etc.
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
    """
    return ["Barbie", "Oppenheimer"]


def find_theaters(location: str, movie: str = ""):
    """Find theaters based on location and optionally movie title which are is currently playing in theaters.

    Args:
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
        movie: Any movie title
    """
    return ["Googleplex 16", "Android Theatre"]


def get_showtimes(location: str, movie: str, theater: str, date: str):
    """
    Find the start times for movies playing in a specific theater.

    Args:
      location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
      movie: Any movie title
      thearer: Name of the theater
      date: Date for requested showtime
    """
    return ["10:00", "11:00"]
    

functions = {
    "find_movies": find_movies,
    "find_theaters": find_theaters,
    "get_showtimes": get_showtimes,
}


model = genai.GenerativeModel(model_name="gemini-1.0-pro", tools=functions.values())

response = model.generate_content(
    "Which theaters in Mountain View show the Barbie movie?"
)

part = response.candidates[0].content.parts[0]


def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)
    
    
if part.function_call:
    result = call_function(part.function_call, functions)

print("Function Result: ", result)

# Put the result in a protobuf Struct
s = Struct()
s.update({"result": result})


function_response = glm.Part(
    function_response=glm.FunctionResponse(name=part.function_call.name, response=s)
)

# Build the message history
messages = [
    # fmt: off
    {"role": "user",
     "parts": ["Which theaters in Mountain View show the Barbie movie?."]},
    {"role": "model",
     "parts": response.candidates[0].content.parts},
    {"role": "user",
     "parts": [function_response]},
    # fmt: on
]

response = model.generate_content(messages)
print(response.text)

