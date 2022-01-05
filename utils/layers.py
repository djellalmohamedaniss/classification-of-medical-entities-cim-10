import typing
from typing import Any
import tensorflow as tf


class DecoderInput(typing.NamedTuple):
    tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any


class BaseDecoder(tf.keras.layers.Layer):

    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(BaseDecoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.output_vocab_size, activation='softmax')

    def call(self, inputs: DecoderInput):
        vectors = self.embedding(inputs.tokens)
        output, state_h, state_c = self.lstm(vectors, initial_state=inputs.enc_output)
        logits = self.dense(output)
        return logits, state_h, state_c


class BaseEncoder(tf.keras.layers.Layer):

    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(BaseEncoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                   embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, tokens, state=None):
        vectors = self.embedding(tokens)
        output, state_h, state_c = self.lstm(vectors, initial_state=state)
        return output, state_h, state_c
