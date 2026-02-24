import tensorflow as tf
import numpy as np
import math
from keras.layers import Layer


# Time embedding layer
class Time_embedding_layer(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len, **kwargs):
        super(Time_embedding_layer, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
        })
        return config


    def get_time_encoding_table(self, max_seq_len, d_model, t_set):
        def reverse_min_max_normalization(normalized_value, min_val, max_val):
            original_value = normalized_value * (max_val - min_val) + min_val
            return original_value

        def cal_angle(el_time, hid_idx):
            el_time = reverse_min_max_normalization(el_time, 0.5, 5)
            return el_time / np.power(max_seq_len, 2 * (hid_idx // 2) / d_model)

        def get_timei_angle_vec(time_value):
            angels = [cal_angle(time_value, hid_j) for hid_j in range(d_model)]
            angels = tf.convert_to_tensor(angels, dtype=tf.float32)
            return angels

        time_encoding_matrix = tf.map_fn(lambda time_i: get_timei_angle_vec(time_i), t_set,
                                         fn_output_signature=tf.float32)

        # Calculate sine of the elements in dim 2i (even indices)
        sin_values = tf.math.sin(time_encoding_matrix[:, 0::2, :])

        # Calculate cosine of the elements in dim 2i+1 (odd indices)
        cos_values = tf.math.cos(time_encoding_matrix[:, 1::2, :])

        # Concatenate sin_values and cos_values and store them in time_encoding_matrix
        time_encoding_matrix = tf.concat([sin_values, cos_values], axis=1)

        return time_encoding_matrix

    def time_encoding(self, time_data):
        batch_size = tf.shape(time_data)[0]
        time_embedding = tf.zeros((batch_size, self.max_seq_len, self.d_model), dtype=tf.float32)
        time_embedding = self.get_time_encoding_table(self.max_seq_len, self.d_model, time_data)
        time_embedding = tf.transpose(time_embedding, perm=[0, 2, 1])
        return time_embedding

    def call(self, data_, time_):
        # Make embeddings relatively larger
        data_ = data_ * math.sqrt(self.d_model)
        time_embedding = self.time_encoding(time_)

        # Rearange the time embedding matrix even, odd... because sin and cos were concatenated previuosly.
        temp_data_even = time_embedding[:, :, 0::2]
        temp_data_odd = time_embedding[:, :, 1::2]
        arranged_time_embedding = tf.concat([temp_data_even, temp_data_odd], axis=-1)

        data_ = data_ + arranged_time_embedding
        return data_, arranged_time_embedding
