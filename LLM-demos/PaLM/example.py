"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""

import google.generativeai as palm
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PALM_API_KEY")

palm.configure(api_key=api_key)

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
print(model)

prompt = """
You are an expert at solving word problems.

Solve the following problem:

I have three houses, each with three cats.
each cat owns 4 mittens, and a hat. Each mitten was
knit from 7m of yarn, each hat from 4m.
How much yarn was needed to make all the items?

Think about it step by step, and show your work.
"""

completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
)

print(completion.result)
