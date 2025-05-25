from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)


class Task(BaseModel):
    task_id: int
    description: str
    assigned_to: str = Field(description="Which worker type should handle this? E.g., Researcher, Writer, Coder")

class Plan(BaseModel):
    goal: str
    steps: list[Task]

user_goal = "Write a short blog post about the benefits of AI agents."

prompt_planner = f"""
Create a step-by-step plan to achieve the following goal. 
Assign each step to a hypothetical worker type (Researcher, Writer).
 
Goal: {user_goal}
"""
 
print(f"Goal: {user_goal}")
print("Generating plan...")

response_plan = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=prompt_planner,
    config={
        "response_mime_type": "application/json",
        "response_schema": Plan
    },
)

for step in response_plan.parsed.steps:
    print(f"Step {step.task_id}: {step.description} (Assignee: {step.assigned_to})")
