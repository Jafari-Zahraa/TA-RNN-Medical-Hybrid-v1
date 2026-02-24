from keras.models import Model
from keras.layers import (
    Input, Embedding, TimeDistributed, Lambda,
    GRU, Bidirectional, Dropout, Dense,
    concatenate, Add, LayerNormalization
)
from keras.regularizers import l2
from keras.layers import MultiHeadAttention
import tensorflow as tf
from TimeEmbeddingLayer import Time_embedding_layer
from AttentionLayer import AttentionLayer
from LossFunction import binary_cross_entropy
from GlobalVariablesMIMIC import time_steps, demographic_features

def TA_RNN_Medical_Hybrid(embedding_matrix, cell, drout, L2,  hidden_s, num_icd_per_visit):

    """
    Paper-Implementation with medical knowledge + Multi-Head Attention + interpretable attention
    """

    embedding_dim = embedding_matrix.shape[1]

    # 1️⃣ Inputs
    icd_input = Input(shape=(time_steps, num_icd_per_visit), name="icd_input")
    time_input = Input(shape=(time_steps,), name="time_input")
    demographic_input = Input(shape=(demographic_features,), name="demo_input")

    # 2️⃣ Medical Embedding
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True,
        name="medical_embedding"
    )
    embedded_codes = TimeDistributed(embedding_layer)(icd_input)  # (B, T, 49, emb_dim)

    # 3️⃣ Visit-level pooling
    visit_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=2), name="visit_pooling")(embedded_codes)

    # 4️⃣ Time Embedding
    time_encoder = Time_embedding_layer(d_model=embedding_dim, max_seq_len=time_steps)
    visit_embedding, _ = time_encoder(visit_embedding, time_input)  # (B, T, emb_dim)

    RNN_1 = Bidirectional(GRU(
            hidden_s,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            activity_regularizer=l2(L2)
        ),
        name="bigru_1"
    )(visit_embedding)

    RNN_1 = Dropout(drout)(RNN_1)

    RNN_2 = Bidirectional(
        GRU(
            hidden_s // 2,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            activity_regularizer=l2(L2)
        ),
        name="bigru_2"
    )(RNN_1)

    RNN_2 = Dropout(drout)(RNN_2)

    # 6️⃣ Multi-Head Self-Attention
    mha = MultiHeadAttention(num_heads=4, key_dim=hidden_s // 4, name="mha_visits")
    mha_out = mha(RNN_2, RNN_2)
    mha_out = Add(name="mha_residual")([RNN_2, mha_out])
    mha_out = LayerNormalization(name="mha_norm")(mha_out)

    # 7️⃣ Project visit_embedding to match hidden_s
    visit_proj = Dense(hidden_s, activation=None, name="visit_proj")(visit_embedding)

    # 8️⃣ Interpretable Attention
    attention_layer = AttentionLayer(hidden_s)
    alpha, beta, context_vector = attention_layer([mha_out, visit_proj])

    # 9️⃣ Demographic Fusion
    fusion = concatenate([context_vector, demographic_input], name="fusion_layer")

    # 🔟 MLP Classifier
    x = Dense(8, activation='relu')(fusion)
    x = Dense(4, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="output")(x)

    # 🔹 Model
    model = Model(inputs=[icd_input, time_input, demographic_input], outputs=output, name="TA_RNN_Medical_Hybrid")

    model.compile(loss=binary_cross_entropy, optimizer='adam', metrics=['accuracy'])

    return model