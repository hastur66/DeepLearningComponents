from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import convert_to_tensor, string
from tensorflow.data import Dataset
from positional_encoding import PositionEmbeddingFixedWeights

output_sequence_length = 5
vocab_size = 10
output_length = 6

sentences = [["I am a robot"], ["you too robot"]]
sentence_data = Dataset.from_tensor_slices(sentences)

vectorize_layer = TextVectorization(
                  output_sequence_length=output_sequence_length,
                  max_tokens=vocab_size)

vectorize_layer.adapt(sentence_data)
word_tensors = convert_to_tensor(sentences, dtype=string)
vectorized_words = vectorize_layer(word_tensors)

# attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_sequence_length,
#                                             vocab_size, output_length)
# attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
# print(attnisallyouneed_output.shape)

def test_call():
    attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_sequence_length,
                                            vocab_size, output_length)
    attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
    assert attnisallyouneed_output.shape == (2, 5, 6)