# -*- coding: utf-8 -*-
# @Time : 2022/3/30 22:50
# @Author : Jclian91
# @File : model.py
# @Place : Minghang, Shanghai
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential


class MultiHeadAttention(Layer):
    """
    # Input
        three 3D tensor: Q, K, V
        each with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input0 steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = MultiHeadAttention(8,16)([embeddings,embeddings,embeddings]) # self Attention
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, key_size=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['key_size'] = self.key_size
        return config

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(units=self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(units=self.out_dim, use_bias=False)

    def call(self, inputs):
        Q_seq, K_seq, V_seq = inputs

        Q_seq = self.q_dense(Q_seq)
        K_seq = self.k_dense(K_seq)
        V_seq = self.v_dense(V_seq)

        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.heads, self.key_size))
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.heads, self.key_size))
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.heads, self.size_per_head))

        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # Attention
        A = tf.einsum('bjhd,bkhd->bhjk', Q_seq, K_seq) / self.key_size ** 0.5
        A = K.softmax(A)

        O_seq = tf.einsum('bhjk,bkhd->bjhd', A, V_seq)
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.out_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class LayerNormalization(Layer):

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config['center'] = self.center
        config['scale'] = self.scale
        config['epsilon'] = self.epsilon
        return config

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

    def call(self, inputs):
        if self.center:
            beta = self.beta
        if self.scale:
            gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TransformerBlock(Layer):
    """
    # Input
        3D tensor: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, input steps, head number * head size)`.
    Note: The layer has been tested with Keras 2.3.1 (Tensorflow 1.14.0 as backend)
    Example:
        S_inputs = Input(shape=(None,), dtype='int32')
        embeddings = Embedding(max_features, 128)(S_inputs)
        result_seq = TransformerBlock(8,16,128)(embeddings)
        result_vec = GlobalMaxPool1D()(result_seq)
        result_vec = Dropout(0.5)(result_vec)
        outputs = Dense(1, activation='sigmoid')(result_vec)
    """

    def __init__(self, heads, size_per_head, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.ff_dim = ff_dim
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config['heads'] = self.heads
        config['size_per_head'] = self.size_per_head
        config['ff_dim'] = self.ff_dim
        config['rate'] = self.rate
        return config

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)
        assert input_shape[-1] == self.heads * self.size_per_head
        self.att = MultiHeadAttention(heads=self.heads, size_per_head=self.size_per_head)
        self.ffn = Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.heads * self.size_per_head),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, inputs):
        attn_output = self.att([inputs, inputs, inputs])
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)