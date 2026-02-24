import tensorflow as tf
# 𝐿𝑜𝑠𝑠 = −1/𝑁 ∑(𝛼 ∙ (𝑦 ∙ 𝑙𝑜𝑔 𝑦′)) + ((1 − 𝛼) ∙ (1 − 𝑦) ∙ 𝑙𝑜𝑔(1 − 𝑦′))
def binary_cross_entropy(y, yhat, epsilon=0.7):
    y = tf.cast(y, tf.float32)
    yhat = tf.cast(yhat, tf.float32)

    loss = -tf.reduce_mean(
        (epsilon * y * tf.math.log(yhat + 1e-6)) +
        ((1.0 - epsilon) * (1 - y) * tf.math.log(1 - yhat + 1e-6)),
        axis=-1
    )
    return loss
