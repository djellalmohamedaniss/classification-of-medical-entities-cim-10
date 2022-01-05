import json

import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer
from transformers import CamembertTokenizer

from data import read_data_all_code, get_vocabulary
from seq2seq import CIMCodeSeq2SeqBert, BertEncoder, load_camembert_base
from utils.losses import MaskedLoss
from utils.preprocessing import transform_cim_code, tf_lower_and_split_cim, \
    no_token_tf_lower_and_split_punctuation

embedding_dim = 64
units = 128

x_train, train_labels, x_test, test_labels, x_val, val_labels, _ = read_data_all_code()

train_labels = [transform_cim_code(code) for code in train_labels]

output_vocabulary = get_vocabulary(train_labels)

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_cim)

output_text_processor.adapt(output_vocabulary)

target_text_processor = tf.keras.layers.TextVectorization(standardize=no_token_tf_lower_and_split_punctuation,
                                                          vocabulary=output_text_processor.get_vocabulary())


output_converter = tf.keras.layers.StringLookup(
            vocabulary=target_text_processor.get_vocabulary(),
            mask_token='',
            invert=True)

seq2seq = CIMCodeSeq2SeqBert(
    embedding_dim, units,
    output_text_processor=output_text_processor,
    target_text_processor=target_text_processor,
    output_converter=output_converter,
    bert_vocab_model=CamembertTokenizer.pretrained_vocab_files_map.get("vocab_file").get("camembert-base"))

# Configure the loss and optimizer
seq2seq.compile(
    optimizer=tf.optimizers.RMSprop(),
    loss=MaskedLoss()
)

dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
dataset = dataset.batch(16, drop_remainder=True)

trained_model = seq2seq.fit(dataset, epochs=10)

with open('./history.json', 'w') as fp:
    json.dump(trained_model.history, fp)

seq2seq.save("seq2seq")
