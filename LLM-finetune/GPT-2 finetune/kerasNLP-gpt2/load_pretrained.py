import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow import keras
import numpy as np
import time


gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset('gpt2_base_en')
gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    'gpt2_base_en',
    sequence_length=256,
    add_end_token=True,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset('gpt2_base_en', preprocessor=gpt2_preprocessor)


start = time.time()

output = gpt2_lm.generate('My trip to Yosemite was', max_length=200)

print("GPT2 Output:\n")
print(output.numpy().decode('utf-8')) 

end = time.time()
print(f"Time taken to generate: {end - start}")
