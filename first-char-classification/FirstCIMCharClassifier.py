import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

from data import read_data_first_char, encode_labels, get_vocabulary
from utils.predictions import generate_class_weights

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
    STEPS:
    - calculate class weights and generate a weight dictionary
    - define a TextVectorization layer with: vocabulary = all words in the "RawText" column
    - an embedding layer that takes the integer-encoded vocabulary and looks up the embedding vector for 
    each word-index. These vectors are learned as the model trains.
    - a GlobalAveragePooling1D layer to return a fixed length vector no matter the sequence length
    - we add 1 dense layer that acts as the output layer with X classes ( depending on the number 
    of classes in our dataset )
"""


x_train, train_labels, x_test, test_labels, x_val, val_labels = read_data_first_char()

# to encode labels, we will use LabelEncoder that transforms string labels into integers.
y_train, y_test, y_val, classes_number, _ = encode_labels(train_labels, test_labels, val_labels)

vocabulary = get_vocabulary(x_train)
vocab_size = len(vocabulary)
embedding_dim = 256

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.

vectorize_layer = TextVectorization(
    standardize='lower_and_strip_punctuation',
    vocabulary=vocabulary,
    output_mode='int')

inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorize_layer(inputs)
x = Embedding(vocab_size, embedding_dim, name="embedding")(x)
x = GlobalAveragePooling1D(name="aggregator")(x)
x = Dense(embedding_dim, activation='relu')(x)
outputs = Dense(classes_number, name="classification")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="first_cim_classifier")

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])

trained_model = model.fit(np.array(x_train), np.array(y_train), batch_size=256, epochs=10,
                          class_weight=generate_class_weights(y_train, classes_number),
                          validation_data=(np.array(x_val), np.array(y_val)))

test_scores = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)

with open('./history.json', 'w') as fp:
    json.dump(trained_model.history, fp)

model.save('./first_cim_classifier.h5')
