from numpy import random
import pytest
from multihead_attention import MultiHeadAttention

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process
 
queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))
 
multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
output = multihead_attention(queries, keys, values)
# print(output)

def test_call():
    assert output.shape == (64, 5, 512)