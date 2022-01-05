import tensorflow as tf


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask
        return tf.reduce_sum(loss)
