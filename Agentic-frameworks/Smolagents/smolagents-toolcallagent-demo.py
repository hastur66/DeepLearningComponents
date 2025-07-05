from smolagents import ToolCallingAgent, InferenceClientModel
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

model = InferenceClientModel(
    model_id=model_id,
    token=HUGGINGFACE_TOKEN,
)

agent = ToolCallingAgent(
    tools=[],
    model=model,
)

agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
