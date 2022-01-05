import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
from utils.preprocessing import stem_remove_stopwords


def read_data_first_char():
    train_df = pd.read_csv("../datasets/train.csv")
    x_train = train_df["RawText"].tolist()

    test_df = pd.read_csv("../datasets/test.csv")
    x_test = test_df["RawText"].tolist()

    val_df = pd.read_csv("../datasets/val.csv")
    x_val = val_df["RawText"].tolist()

    train_labels = [code[0] for code in train_df["ICD10"].tolist()]
    test_labels = [code[0] for code in test_df["ICD10"].tolist()]
    val_labels = [code[0] for code in val_df["ICD10"].tolist()]

    return x_train, train_labels, x_test, test_labels, x_val, val_labels


def read_data_all_code():
    train_df = pd.read_csv("../datasets/train.csv")
    x_train = train_df["RawText"].tolist()

    test_df = pd.read_csv("../datasets/test.csv")
    x_test = test_df["RawText"].tolist()

    val_df = pd.read_csv("../datasets/val.csv")
    x_val = val_df["RawText"].tolist()

    all_data = list(itertools.chain(x_train, x_test, x_val))

    return x_train, train_df["ICD10"].tolist(), x_test, test_df["ICD10"].tolist(), x_val, val_df["ICD10"].tolist(), len(
        max(all_data, key=len))


def encode_labels(train_labels, test_labels, val_labels):
    all_labels = list(itertools.chain(train_labels, test_labels, val_labels))
    le = LabelEncoder()
    le.fit(all_labels)

    output_classes = list(set(all_labels))
    classes_number = len(output_classes)

    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)
    y_val = le.transform(val_labels)

    return y_train, y_test, y_val, classes_number, le


def get_vocabulary(data: list[str], separator=' ', processing=False) -> list[str]:
    if processing:
        data = [stem_remove_stopwords(sentence) for sentence in data]
    vocabulary_list = []
    for sentence in data:
        vocabulary_list.extend([token.lower().strip() for token in sentence.split(separator) if len(token.strip()) > 0])
    return list(set(vocabulary_list))
