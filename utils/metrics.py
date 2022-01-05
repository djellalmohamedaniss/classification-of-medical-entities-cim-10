import tensorflow as tf


class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.masked_accuracies = self.add_weight(name='mac', initializer='zeros')
        self.masked_count = self.add_weight(name='mc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracies = tf.equal(tf.cast(y_true, dtype=tf.int64), tf.argmax(y_pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        result = tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
        self.masked_accuracies.assign_add(result)
        self.masked_count.assign_add(1)

    def result(self):
        return self.masked_accuracies / self.masked_count


class MaskedLoss(tf.keras.metrics.Metric):

    def __init__(self, name='masked_loss', **kwargs):
        super(MaskedLoss, self).__init__(name=name, **kwargs)
        self.masked_losses = self.add_weight(name='ml', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask
        result = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        self.masked_losses.assign_add(result)
        self.count.assign_add(1)

    def result(self):
        return self.masked_losses / self.count


class BLEUScore(tf.keras.metrics.Metric):

    def __init__(self, name='bleu_score', **kwargs):
        super(BLEUScore, self).__init__(name=name, **kwargs)
        self.bleu_scores = self.add_weight(name='ml', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracies_in_letters = tf.equal(y_true, y_pred)
        accuracies = tf.cast(accuracies_in_letters, dtype=tf.float32)
        bleu_scores = tf.map_fn(lambda row: tf.tensordot(row, tf.constant([0, 0.5, 0.25, 0.125, 0.125, 0]), 1),
                                elems=accuracies)
        self.bleu_scores.assign_add(tf.reduce_mean(bleu_scores))
        self.count.assign_add(1)

    def result(self):
        return self.bleu_scores / self.count


class PositionalCharAccuracy(tf.keras.metrics.Metric):

    def __init__(self, position=1, **kwargs):
        name = str(position) + "_char_accuracy"
        super(PositionalCharAccuracy, self).__init__(name=name, **kwargs)
        self.position = position
        self.scores = self.add_weight(name='fc_scores', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_chars = tf.gather(y_true, tf.range(1, self.position + 1), axis=-1)
        y_pred_chars = tf.gather(y_pred, tf.range(1, self.position + 1), axis=-1)
        accuracies_in_letters = tf.equal(y_true_chars, y_pred_chars)
        reduced_accuracies = tf.reduce_all(accuracies_in_letters, axis=-1)
        accuracies = tf.cast(reduced_accuracies, dtype=tf.float32)
        self.scores.assign_add(tf.reduce_mean(accuracies))
        self.count.assign_add(1)

    def result(self):
        return self.scores / self.count
