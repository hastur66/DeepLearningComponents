from transformers import pipeline


## Without specifying model

# sentiment analysis

classifier = pipeline("sentiment-analysis")

classifier("I've been waiting for a HuggingFace course my whole life.")

classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this!"])


# text generation

generator = pipeline("text-generation")

generator("In this course, we will teach about machine")


# fill-mask

unmasker = pipeline("fill-mask")

unmasker("This course will teach you all about <mask> model.", top_k=2)


# named-entity-recognition (NER)

ner = pipeline("ner", grouped_entities=True)

ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


# question answering

question_answer = pipeline("question-answering")

question_answer(
    question="Where do you work",
    context="I work in new york"
)


# summarization

summarizer = pipeline("summarization")

summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
)



## With specifying model


# Text-generation with distilgpt2

generator = pipeline("text-generation", model="distilgpt2")

generator(
    "In this course, we will teach about machine",
    max_length=25,
    num_return_sequences=2
)


## Translation

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

translator("Ce cours est produit par Hugging Face.")

