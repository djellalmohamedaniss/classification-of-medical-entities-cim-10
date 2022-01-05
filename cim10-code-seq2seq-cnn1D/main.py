import json

import tensorflow as tf
import tensorflow_text as tf_text

from data import read_data_all_code, get_vocabulary
from seq2seq_cnn1D import CIMCodeSeq2Seq1DCNN
from utils.losses import MaskedLoss
from utils.preprocessing import tf_lower_and_split_punctuation, tf_lower_and_split_cim, \
    no_token_tf_lower_and_split_punctuation, SimpleTokenizer, \
    transform_cim_code

physical_devices = tf.config.list_physical_devices('GPU')

embedding_dim = 256
units = 512

BATCH_SIZE = 128

x_train, train_labels, x_test, test_labels, x_val, val_labels, _ = read_data_all_code()

train_labels = [transform_cim_code(code) for code in train_labels]

input_vocabulary = get_vocabulary(x_train)
output_vocabulary = get_vocabulary(train_labels)

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punctuation)

input_text_processor.adapt(input_vocabulary)

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_cim)

output_text_processor.adapt(output_vocabulary)

target_text_processor = tf.keras.layers.TextVectorization(standardize=no_token_tf_lower_and_split_punctuation,
                                                          vocabulary=output_text_processor.get_vocabulary())

output_converter = tf.keras.layers.StringLookup(
    vocabulary=target_text_processor.get_vocabulary(),
    mask_token='',
    invert=True)

index_from_string = tf.keras.layers.StringLookup(
    vocabulary=output_text_processor.get_vocabulary(), mask_token='')

seq2seq = CIMCodeSeq2Seq1DCNN(
    embedding_dim, units,
    input_text_processor=input_text_processor,
    target_text_processor=target_text_processor,
    output_text_processor=output_text_processor,
    char_tokenizer=tf_text.UnicodeCharTokenizer(),
    word_tokenizer=SimpleTokenizer(),
    output_converter=output_converter,
    filters=64,
    kernel_size=1,
    batch_size=BATCH_SIZE)

# Configure the loss and optimizer
seq2seq.compile(
    optimizer=tf.optimizers.RMSprop(),
    loss=MaskedLoss()
)

dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

trained_model = seq2seq.fit(dataset, epochs=10)

with open('./history.json', 'w') as fp:
    json.dump(trained_model.history, fp)

