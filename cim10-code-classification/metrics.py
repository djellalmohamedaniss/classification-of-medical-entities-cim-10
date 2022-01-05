import tensorflow as tf


class CharAccuracy(tf.keras.metrics.Metric):

    def __init__(self, le: tf.lookup.StaticHashTable, name='char_accuracy', k=0, length=1, **kwargs):
        super(CharAccuracy, self).__init__(name=name, **kwargs)
        self.le = le
        self.k = k
        self.length = length
        self.true_chars = self.add_weight(name='tc', initializer='zeros')
        self.total_count = self.add_weight(name='tc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        labeled_y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        y_true = tf.cast(y_true, dtype=tf.int64)
        cim_true_label = tf.map_fn(lambda code: tf.strings.substr(code, self.k, self.length),
                                   elems=tf.reshape(self.le.lookup(y_true), shape=(-1,)))
        cim_pred_label = tf.map_fn(lambda code: tf.strings.substr(code, self.k, self.length),
                                   elems=self.le.lookup(labeled_y_pred))
        scores = tf.cast(tf.equal(cim_pred_label, cim_true_label), dtype=tf.float32)
        self.true_chars.assign_add(tf.reduce_mean(scores))
        self.total_count.assign_add(1)

    def result(self):
        return self.true_chars / self.total_count


class BLEUScore(tf.keras.metrics.Metric):

    def __init__(self, le: tf.lookup.StaticHashTable, bleu_weights=None, name='bleu_score_3_char', **kwargs):
        super(BLEUScore, self).__init__(name=name, **kwargs)
        if bleu_weights is None:
            self.bleu_weights = [0.25, 0.25, 0.25]
        else:
            self.bleu_weights = bleu_weights
        self.le = le
        self.true_chars = self.add_weight(name='tc', initializer='zeros')
        self.total_count = self.add_weight(name='tc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        labeled_y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
        y_true = tf.cast(y_true, dtype=tf.int64)
        cim_true_label = tf.map_fn(lambda code: tf.strings.bytes_split(tf.strings.substr(code, 0, 3)),
                                   elems=tf.reshape(self.le.lookup(y_true), shape=(-1,)))
        cim_pred_label = tf.map_fn(lambda code: tf.strings.bytes_split(tf.strings.substr(code, 0, 3)),
                                   elems=self.le.lookup(labeled_y_pred))
        scores = tf.cast(tf.equal(cim_pred_label, cim_true_label), dtype=tf.float32)
        bleu_score = tf.map_fn(lambda row: tf.tensordot(row, tf.constant(self.bleu_weights), 1),
                               elems=scores)
        self.true_chars.assign_add(tf.reduce_mean(bleu_score))
        self.total_count.assign_add(1)

    def result(self):
        return self.true_chars / self.total_count
