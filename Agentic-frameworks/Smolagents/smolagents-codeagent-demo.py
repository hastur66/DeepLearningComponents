from smolagents import CodeAgent, HfApiModel
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

model = HfApiModel(
    model_id=model_id,
    token=HUGGINGFACE_TOKEN,
)

agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["requests", "bs4"]
)

agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
