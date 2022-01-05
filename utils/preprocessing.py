import tensorflow as tf
import tensorflow_text as tf_text
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords

stemmer = FrenchStemmer()
french_stopwords = stopwords.words('french')


def tf_lower_and_split_punctuation(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z0-9.?!¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def tf_lower_and_split_punctuation_no_tokens(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z0-9.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    return text


def no_token_tf_lower_and_split_punctuation(text):
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text], separator=' ')
    return text


def tf_lower_and_split_cim(text):
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def tf_lower_and_split_chars(text):
    text = tf.strings.lower(text)
    text = tf.strings.strip(text)
    split_text = tf.strings.bytes_split(text)
    text = tf.strings.reduce_join(split_text, separator=" ", axis=-1)
    return text


def get_cim_vocabulary(data: list[str]) -> list[str]:
    vocabulary_list = []
    for sentence in data:
        vocabulary_list.extend([token.lower().strip() for token in list(sentence) if len(token.strip()) > 0])
    return list(set(vocabulary_list))


def get_char_vocabulary(data):
    vocabulary_list = []
    for sentence in data:
        vocabulary_list.extend(
            [token.lower().strip() for token in list("".join(list(sentence))) if len(token.strip()) > 0])
    return list(set(vocabulary_list))


def transform_cim_code(code):
    return " ".join(list(code))


def stem_remove_stopwords(sentence):
    # removed_stopwords = [word for word in stemmed_sentence if word not in french_stopwords]
    stemmed_sentence = [stemmer.stem(word) for word in list(sentence)]
    return " ".join(stemmed_sentence)


class SimpleTokenizer(tf_text.Tokenizer):
    def tokenize(self, input):
        text = tf.strings.lower(input)
        text = tf.strings.regex_replace(text, '[^ a-z0-9.?!¿]', '')
        text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
        return tf.strings.split(text)
