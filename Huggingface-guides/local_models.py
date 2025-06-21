from transformers import pipeline


# Task 1 - Text generation
generator = pipeline("text-generation", model="distilgpt2")

response = generator(
    "In this coures we will teach you",
    max_length=30,
    num_return_sequences=2,
)

print(response)