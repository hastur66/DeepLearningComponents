from anthropic import AnthropicBedrock
import os
from dotenv import load_dotenv

load_dotenv()


client = AnthropicBedrock(
    aws_access_key="ACCESS-KEY",
    aws_secret_key="SECRET-KEY",
    # aws_session_token="SESSION_TOKEN",
    aws_region="us-west-2",
)

message = client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello, world"}]
)

print(message.content)
