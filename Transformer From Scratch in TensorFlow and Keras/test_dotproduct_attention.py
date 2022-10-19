from numpy import random
import pytest
from dotproduct_attention import DotProductAttention

d_k = 64
d_v = 64
batch_size = 64
input_seq_length = 5

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()
output = attention(queries, keys, values, d_k)

# class TestDotProductAttention():
def test_call():
    assert output.shape == (64, 5, 64)