from collections import Counter

import tensorflow as tf
import numpy as np


def evaluate(model, sentence, output_text_processor, max_length=5):
    output_converter = tf.keras.layers.StringLookup(
        vocabulary=output_text_processor.get_vocabulary(),
        mask_token='',
        invert=True)

    target = ""

    for _ in range(max_length):
        letters = tf.argmax(model.predict([np.array([sentence]), np.array([target])]), -1)
        new_letter = output_converter(letters).numpy()[0][0].decode()
        if new_letter == "[END]":
            return "".join(list(target))
        target = " ".join([target, new_letter])
    return "".join(list(target)).upper()


def generate_class_weights(labels, number_of_labels):
    count_labels = Counter(labels)
    total = len(labels)
    weights = {}
    uniq_labels = list(set(labels))
    factor = total / number_of_labels
    for label in uniq_labels:
        weights[label] = factor / count_labels[label]
    return weights
