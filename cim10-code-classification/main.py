import json

import numpy as np
import tensorflow as tf
from keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import TextVectorization, Embedding, Dense

from data import read_data_all_code, encode_labels, get_vocabulary
from metrics import CharAccuracy, BLEUScore

x_train, train_labels, x_test, test_labels, x_val, val_labels, _ = read_data_all_code()

# to encode labels, we will use LabelEncoder that transforms string labels into integers.
y_train, y_test, y_val, classes_number, le = encode_labels(train_labels, test_labels, val_labels)

keys_tensor = tf.constant(y_train)
values_tensor = tf.constant(train_labels)
ht = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
    default_value="....")

vocabulary = get_vocabulary(x_train)
vocab_size = len(vocabulary)
embedding_dim = 256

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.

vectorize_layer = TextVectorization(
    standardize='lower_and_strip_punctuation',
    vocabulary=vocabulary,
    output_mode='int')

old_model = tf.keras.models.load_model("../first-char-classification/first_cim_classifier-lstm.h5")

inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorize_layer(inputs)
x = Embedding(vocab_size, embedding_dim, name="embedding", trainable=False,
              weights=old_model.layers[2].get_weights())(x)
x = Bidirectional(LSTM(512))(x)
x = Dense(2048)(x)
outputs = Dense(classes_number, name="classification")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="first_cim_classifier")

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy", CharAccuracy(ht), CharAccuracy(ht, length=2, name='2_char_accuracy'), CharAccuracy(ht, length=3, name='3_char_accuracy'),
             BLEUScore(ht, bleu_weights=[0.5, 0.3, 0.2], name='weighted_bleu_score')])

trained_model = model.fit(np.array(x_train), np.array(y_train), batch_size=256, epochs=10)

# test_scores = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)

with open('./history.json', 'w') as fp:
    json.dump(trained_model.history, fp)

model.save('./cim10-code-classifier.h5')
