from autogen import ConversableAgent
import os


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": OPENAI_API_KEY}]},
    human_input_mode="NEVER",  # Never ask for human input.
)


joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": OPENAI_API_KEY}]},
    human_input_mode="NEVER",  # Never ask for human input.
)


result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)
print(result)