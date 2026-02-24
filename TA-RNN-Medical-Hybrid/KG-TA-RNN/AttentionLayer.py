import tensorflow as tf
from keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[1][-1]

        self.W_alpha = self.add_weight(name="W_alpha", shape=(self.units, 1),
                                       initializer="glorot_uniform", trainable=True)
        self.b_alpha = self.add_weight(name="b_alpha", shape=(1,),
                                       initializer="zeros", trainable=True)

        self.W_beta = self.add_weight(name="W_beta", shape=(self.units, feature_dim),
                                      initializer="glorot_uniform", trainable=True)
        self.b_beta = self.add_weight(name="b_beta", shape=(feature_dim,),
                                      initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        h, input_data = inputs

        # Visit-level attention (alpha)
        e = tf.matmul(h, self.W_alpha) + self.b_alpha
        e = tf.squeeze(e, axis=-1)
        alpha_unmasked = tf.nn.softmax(e, axis=-1)
        alpha = tf.expand_dims(alpha_unmasked, axis=-1)

        # Feature-level attention (beta)
        beta_unmasked = tf.tanh(tf.matmul(h, self.W_beta) + self.b_beta)
        beta = tf.nn.softmax(beta_unmasked, axis=-1)

        # Mask
        if mask is not None:
            mask_broadcasted = tf.expand_dims(mask[0], axis=-1)
            alpha *= tf.cast(mask_broadcasted, tf.float32)
            alpha /= tf.reduce_sum(alpha, axis=1, keepdims=True)
            input_data *= tf.cast(mask_broadcasted, tf.float32)

        # Context vector
        c = tf.reduce_sum(alpha * beta * input_data, axis=1)

        return alpha, beta, c

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        time_steps = input_shape[0][1]
        feature_dim = input_shape[1][-1]
        return [(batch_size, time_steps, 1),
                (batch_size, time_steps, feature_dim),
                (batch_size, feature_dim)]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
