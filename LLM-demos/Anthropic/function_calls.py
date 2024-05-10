from anthropic import Anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

client = Anthropic(
	api_key=API_KEY,
	)

def get_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


messages=[{"role": "user", "content": "What is the weather like in San Francisco?"}]

tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature, either \"celsius\" or \"fahrenheit\""
                        }
                    },
                    "required": ["location"]
                }
            }
        ]


def run_tool(messages, tools):
    response = client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    response_message = response.content
    tool_calls = response.content[1]

    if tool_calls:
        available_functions = {
            "get_weather": get_weather,
        }

        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.content[0].text,
                },
                {
                    "type": "tool_use",
                    "id": tool_calls.id,
                    "name": tool_calls.name,
                    "input": tool_calls.input,
                }
            ]
        })  
        
        function_name = tool_calls.name
        function_to_call = available_functions[function_name]
        function_response = function_to_call(
            location=tool_calls.input.get("location"),
            unit=tool_calls.input.get("unit"),
        )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_calls.id,
                        "content": function_response
                    }
                ]
            }
        )

        second_response = client.beta.tools.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=messages,
            tools=tools
        ) 
        return second_response


    print(second_response)

print(run_tool(messages, tools))
