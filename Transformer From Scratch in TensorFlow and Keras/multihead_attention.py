import tensorflow as tf
from tensorflow import matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(cast(d_k, float32))
        
        if mask is not None:
            scores += -1e9 * mask

        weight = softmax(scores)

        return matmul(weight, values)

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.head = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_v)
        self.W_o = Dense(d_model)

    def reshape_tensor(self, x, head, flag):
        if flag:
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], head, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.head, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.head, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.head, True)

        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        
        output = self.reshape_tensor(o_reshaped, self.head, False)

        return self.W_o(output)
