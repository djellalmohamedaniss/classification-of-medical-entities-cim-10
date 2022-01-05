from typing import Any

import tensorflow as tf

from utils.layers import DecoderInput, BaseEncoder, DecoderOutput
from utils.metrics import MaskedAccuracy, MaskedLoss, BLEUScore, PositionalCharAccuracy


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        w1_query = self.W1(query)
        w2_key = self.W2(value)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):

    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(self.dec_units)
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)
        self.fc = tf.keras.layers.Dense(self.output_vocab_size, activation='softmax')

    def call(self, inputs: DecoderInput, state=None) -> tuple[DecoderOutput, Any, Any]:
        vectors = self.embedding(inputs.tokens)
        output, state_h, state_c = self.lstm(vectors, initial_state=state)

        context_vector, attention_weights = self.attention(query=output, value=inputs.enc_output, mask=inputs.mask)
        context_and_rnn_output = tf.concat([context_vector, output], axis=-1)
        attention_vector = self.Wc(context_and_rnn_output)

        logits = self.fc(attention_vector)
        return DecoderOutput(logits, attention_weights), state_h, state_c


class CIMCodeSeq2SeqAttention(tf.keras.Model):

    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor, target_text_processor,
                 output_converter, batch_size):
        super().__init__()
        encoder = BaseEncoder(input_text_processor.vocabulary_size(),
                              embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                          embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size

        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.target_text_processor = target_text_processor
        self.output_converter = output_converter

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
                batch_loss, dec_state_h, dec_state_c = self._loop_step(new_tokens, input_mask, enc_output,
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
        return metrics_dict

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state_h, dec_state_c):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(tokens=input_token,
                                     enc_output=enc_output,
                                     mask=input_mask)

        decoder_output, dec_state_h, dec_state_c = self.decoder(decoder_input, state=[dec_state_h, dec_state_c])

        y = target_token
        y_pred = decoder_output.logits
        loss = self.loss(y, y_pred)
        self.masked_accuracy.update_state(y, y_pred)
        self.masked_loss.update_state(y, y_pred)
        return loss, dec_state_h, dec_state_c

    def _preprocess(self, input_text, target_text):
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0
        return input_tokens, input_mask, target_tokens, target_mask

    def call(self, inputs):

        input_text, target_text = inputs
        input_tokens, input_mask, _, target_mask = self._preprocess(input_text, target_text)
        target_tokens = self.target_text_processor(target_text)
        enc_output, dec_state_h, dec_state_c = self.encoder(input_tokens)
        max_target_length = tf.shape(target_tokens)[1]

        for t in tf.range(max_target_length):
            input_token = target_tokens[:, t:t + 1]
            decoder_input = DecoderInput(tokens=input_token,
                                         enc_output=enc_output,
                                         mask=input_mask)
            decoder_output, dec_state_h, dec_state_c = self.decoder(decoder_input, state=[dec_state_h, dec_state_c])

        return decoder_output.logits

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
