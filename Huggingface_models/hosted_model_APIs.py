import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def query(payload, API_URL):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# Task 1 - Fill Mask task
payload = {"inputs": "The answer to the universe is [MASK]"}

def fill_mask_task(payload):
    """ Tries to fill in a hole with a missing word (token to be precise). That’s the base task for BERT models.
      """    
    
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
    response = query(payload, API_URL)
    print(response)
    return response[0]["sequence"]


# Task 2 - Summarization task
payload = {
        "inputs": """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building,
                and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.
                During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
                man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City 
                was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a 
                broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres 
                (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after 
                the Millau Viaduct.""",
        "parameters": {"do_sample": False},
        }

def summarization_task(payload):
    """ Summarizes a text.
      """
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    response = query(payload, API_URL)
    print(response)
    return response[0]["summary_text"]


# Task 3 - Question Answering task
payload = {"input": 
            {
            "question": "What is the capital of France?",
            "context": "The capital of France is Paris."
            }
           }

def question_answering_task(payload):
    """ Answers a question about a context.
      """
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    response = query(payload, API_URL)
    print(response)
    return response


# Task 4 - Table Question Answering task
payload = {
            "inputs": {
            "query": "How many stars does the transformers repository have?",
            "table": {
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": [
                    "Python",
                    "Python",
                    "Rust, Python and NodeJS",
                ],
            },
        }
    }

def table_question_answering_task(payload):
    """ Answers a question about a table.
      """
    API_URL = "https://api-inference.huggingface.co/models/google/tapas-base-finetuned-wtq"
    response = query(payload, API_URL)
    print(response)
    return 


# Task 5 - Sentence Similarity task
payload = {
            "inputs": {
            "source_sentence": "That is a happy person",
            "sentences": ["That is a happy dog", "That is a very happy person", "Today is a sunny day"],
        }
    }

def sentence_similarity_task(payload):
    """ Compares similarity between two sentences.
      """
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    response = query(payload, API_URL)
    print(response)
    return response


# Task 6 - Text Classification task
payload = {"inputs": "I like you. I love you"}

def text_classification_task(payload):
    """ Classifies a text into one of the categories.
      """
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    response = query(payload, API_URL)
    print(response)
    return response

# Task 7 - Text Generation task
#https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task