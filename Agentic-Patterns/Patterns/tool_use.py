from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)


temperature_function = {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco",
            },
        },
        "required": ["location"],
    },
}

# Placeholder function to simulate API call
def get_current_temperature(location: str) -> dict:
    return {"temperature": "15", "unit": "Celsius"}

tools = types.Tool(function_declarations=[temperature_function])

contents = ["What's the temperature in London right now?"]

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=contents,
    config=types.GenerateContentConfig(tools=[tools])
)

response_part = response.candidates[0].content.parts[0]

if response_part.function_call:
    function_call = response_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {dict(function_call.args)}")
 
    # Execute the Function
    if function_call.name == "get_current_temperature":        
        # Call the actual function
        api_result = get_current_temperature(*function_call.args)
        # Append function call and result of the function execution to contents
        follow_up_contents = [
            types.Part(function_call=function_call),
            types.Part.from_function_response(
                name="get_current_temperature",
                response=api_result
            )
        ]
        # Generate final response
        response_final = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents + follow_up_contents,
            config=types.GenerateContentConfig(tools=[tools])
        )
        print(response_final.text)
    else:
        print(f"Error: Unknown function call requested: {function_call.name}")
else:
    print("No function call found in the response.")
    print(response.text)
