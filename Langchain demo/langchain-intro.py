import os
from langchain import LLMChain, PromptTemplate, HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

#initlize hugging face model
flan_t5 = HuggingFaceHub(
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# build template for question answering
template = """Question: {question}
            Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

def single_question(question):
    """single question answering"""
    return print(llm_chain.run(question))

def batch_questions(questions):
    """mutiple question answering"""
    return print(llm_chain.generate(questions))



# single question example
question = "Which NFL team won the Super Bowl in the 2010 season?"
# question = input("What is your question? ")

single_question(question)



# multi question example
questions = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]

batch_questions(questions)