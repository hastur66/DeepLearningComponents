from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
import enum
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)


class EvaluationStatus(enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
 
class Evaluation(BaseModel):
    evaluation: EvaluationStatus
    feedback: str
    reasoning: str


# --- Initial Generation Function ---
def generate_poem(topic: str, feedback: str = None) -> str:
    prompt = f"Write a short, four-line poem about {topic}."
    if feedback:
        prompt += f"\nIncorporate this feedback: {feedback}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    poem = response.text.strip()
    print(f"Generated Poem:\n{poem}")
    return poem


# --- Evaluation Function ---
def evaluate(poem: str) -> Evaluation:
    print("\n--- Evaluating Poem ---")
    prompt_critique = f"""Critique the following poem. 
    Does it rhyme well? 
    Is it exactly four lines? 
    Is it creative? Respond with PASS or FAIL and provide feedback.
    
    Poem:
    {poem}
    """

    response_critique = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_critique,
        config={
            "response_mime_type": "application/json",
            "response_schema": Evaluation
        }
    )

    critique = response_critique.parsed
    print(f"Evaluation Status: {critique.evaluation}")
    print(f"Evaluation Feedback: {critique.feedback}")
    return critique

# Reflection Loop   
max_iterations = 3
current_iteration = 0
topic = "a robot learning to paint"

# simulated poem which will not pass the evaluation
current_poem = "With circuits humming, cold and bright,\nA metal hand now holds a brush"

while current_iteration < max_iterations:
    current_iteration += 1
    print(f"\n--- Iteration {current_iteration} ---")
    evaluation_result = evaluate(current_poem)
 
    if evaluation_result.evaluation == EvaluationStatus.PASS:
        print("\nFinal Poem:")
        print(current_poem)
        break
    else:
        current_poem = generate_poem(topic, feedback=evaluation_result.feedback)
        if current_iteration == max_iterations:
            print("\nMax iterations reached. Last attempt:")
            print(current_poem)
