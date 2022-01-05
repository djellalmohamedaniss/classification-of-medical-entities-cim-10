from nltk import word_tokenize


def tokens_count(sentence: str) -> int:
    return len(word_tokenize(sentence))
