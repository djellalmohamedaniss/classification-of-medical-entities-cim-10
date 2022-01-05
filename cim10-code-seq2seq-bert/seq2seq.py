import typing
from typing import Any

import requests
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer
from transformers import TFCamembertModel

from utils.metrics import MaskedAccuracy, MaskedLoss, BLEUScore, PositionalCharAccuracy


class DecoderInput(typing.NamedTuple):
    tokens: Any
    enc_output: Any
    mask: Any


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self, enc_units):
        super(BertEncoder, self).__init__()
        self.enc_units = enc_units
        camembert_model = TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
        self.embedding = camembert_model
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, tokens, state=None):
        outputs = self.embedding(tokens)
        output, state_h, state_c = self.lstm(outputs.last_hidden_state, initial_state=state)
        return output, state_h, state_c


class Decoder(tf.keras.layers.Layer):

    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
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


def load_camembert_base(link):
    return requests.get(link).content


class CIMCodeSeq2SeqBert(tf.keras.Model):

    def __init__(self, embedding_dim, units, output_text_processor, bert_vocab_model, output_converter,
                 target_text_processor):
        super().__init__()
        encoder = BertEncoder(units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                          embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = 16

        self.output_text_processor = output_text_processor
        self.target_text_processor = target_text_processor
        self.output_converter = output_converter
        self.input_text_processor = SentencepieceTokenizer(model=load_camembert_base(bert_vocab_model))

        # we define metrics
        self.masked_accuracy = MaskedAccuracy()
        self.masked_loss = MaskedLoss()
        self.bleu_score = BLEUScore()
        self.first_char_accuracy = PositionalCharAccuracy()
        self.second_char_accuracy = PositionalCharAccuracy(position=2)
        self.third_char_accuracy = PositionalCharAccuracy(position=3)

    @property
    def metrics(self):
        return [self.masked_loss, self.masked_accuracy, self.bleu_score, self.first_char_accuracy,
                self.second_char_accuracy, self.third_char_accuracy]

    def train_step(self, inputs):
        return self._train_step(inputs)

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                   tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, dec_state_h, dec_state_c = self.encoder(input_tokens)
            loss = tf.constant(0.0)
            for t in tf.range(max_target_length - 1):
                new_tokens = target_tokens[:, t:t + 2]
                batch_loss, dec_state_h, dec_state_c = self._loop_step(new_tokens, input_mask,
                                                                       dec_state_h, dec_state_c)
                loss = loss + batch_loss

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        gradients = tape.gradient(average_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        predictions = self.output_text_processor(self.predict_on_training(input_text, self.batch_size))
        self.bleu_score.update_state(target_tokens, predictions)
        self.first_char_accuracy.update_state(target_tokens, predictions)
        self.second_char_accuracy.update_state(target_tokens, predictions)
        self.third_char_accuracy.update_state(target_tokens, predictions)

        metrics_dict = {m.name: m.result() for m in self.metrics}
        metrics_dict["batch_loss"] = average_loss
        return metrics_dict

    def _loop_step(self, new_tokens, input_mask, dec_state_h, dec_state_c):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(tokens=input_token,
                                     enc_output=[dec_state_h, dec_state_c],
                                     mask=input_mask)

        logits, dec_state_h, dec_state_c = self.decoder(decoder_input)

        y = target_token
        y_pred = logits
        loss = self.compiled_loss(y, y_pred)
        self.masked_accuracy.update_state(y, y_pred)
        self.masked_loss.update_state(y, y_pred)
        return loss, dec_state_h, dec_state_c

    def _preprocess(self, input_text, target_text):
        target_tokens = self.output_text_processor(target_text)
        input_tokens = self.input_text_processor.tokenize(input_text)
        input_tokens_padded = input_tokens.to_tensor(0)
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0
        return input_tokens_padded, input_mask, target_tokens, target_mask

    def call(self, inputs):

        input_text, target_text = inputs
        input_tokens, _, _, _ = self._preprocess(input_text, target_text)
        target_tokens = self.target_text_processor(target_text)
        enc_output, dec_state_h, dec_state_c = self.encoder(input_tokens)
        max_target_length = tf.shape(target_tokens)[1]

        for t in tf.range(max_target_length):
            input_token = target_tokens[:, t:t + 1]
            decoder_input = DecoderInput(tokens=input_token,
                                         enc_output=[dec_state_h, dec_state_c],
                                         mask=None)
            logits, dec_state_h, dec_state_c = self.decoder(decoder_input)

        return logits

    def predict_on_training(self, inputs, batch_size):
        input_text = inputs
        prediction_strings = tf.constant("", shape=(batch_size, 1), dtype=tf.string)
        for i in range(4):
            if i == 0:
                predictions = tf.argmax(self.call([input_text, prediction_strings]), -1)
                prediction_strings = self.output_converter(predictions)
            else:
                predictions = tf.argmax(self.call([input_text, tf.strings.reduce_join(prediction_strings, axis=-1, separator=" ")]),
                                        -1)
                prediction_strings = tf.concat([prediction_strings, self.output_converter(predictions)], axis=-1)
        return tf.strings.reduce_join(prediction_strings, axis=-1, separator=" ")
