from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Permute, Activation
from tensorflow.keras.callbacks import Callback


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], input_shape[-1], 1), initializer='zeros')

        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()


class SelfAttention(Layer):

    def __init__(self, size, num_hops=8, use_penalise=True, penalty_coeff=0.1, **kwargs):
        self.size = size
        self.num_hops = num_hops
        self.use_penalise = use_penalise
        self.penalty_coeff = penalty_coeff
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalise'] = self.use_penalise
        base_config['penalty_coeff'] = self.penalty_coeff
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(self.size, input_shape[2]),
                                  initializer='glorot_uniform', trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(self.num_hops, self.size),
                                  initializer='glorot_uniform', trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)
        attention_score = tf.matmul(W1, hidden_states_transposed)
        attention_score = Activation('tanh')(attention_score)
        attention_weights = tf.matmul(W2, attention_score)
        attention_weights = Activation('softmax')(attention_weights)
        embedding_matrix = tf.matmul(attention_weights, inputs)
        #embedding_matrix_flattened = layers.Flatten()(embedding_matrix)

        if self.use_penalise:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)
            product = tf.matmul(attention_weights, attention_weights_transposed)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0], ))
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product-identity)))
            self.add_loss(self.penalty_coeff * frobenius_norm)

        return embedding_matrix
    

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, maxlen=10):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.maxlen = maxlen
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")

        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, self.maxlen, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, self.maxlen, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


def CNN(vocab_size=10, dimension=10, max_length=10, initial_bias=None, embedding_matrix=None):

    inputs = layers.Input(shape=(max_length,))
    embedding = layers.Embedding(vocab_size,
                                 dimension,
                                 weights=[embedding_matrix],
                                 trainable=False)(inputs)
    x1 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(embedding)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(embedding)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x3 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(embedding)
    x3 = layers.GlobalMaxPooling1D()(x3)

    merged1 = tf.concat([x1, x2], axis=1)
    drop1 = layers.Dropout(0.5)(merged1)

    dense = layers.Dense(64, activation='relu')(drop1)
    drop2 = layers.Dropout(0.36)(dense)

    outputs = layers.Dense(2,
                          activation='softmax',
                          bias_initializer=initial_bias)(drop2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def BLSTM(embedding_model='glove', vocab_size=10, dimension=10, max_length=10, embedding_matrix=10):

    inputs = layers.Input(shape=(max_length, ))

    if embedding_model == 'glove' or embedding_model == 'word2vec':
        embedding = layers.Embedding(vocab_size, dimension, weights=[embedding_matrix])(inputs)

    elif embedding_model == 'bert':
        embedding = layers.Embedding(vocab_size,
                                     dimension,
                                     trainable=False,
                                     embeddings_initializer='uniform')(inputs)

    blstm = layers.Bidirectional(layers.LSTM(200,
                                             return_sequences=True,
                                             activation='tanh',
                                             recurrent_activation='sigmoid',
                                             dropout=0.44,
                                             recurrent_dropout=0.14,
                                             kernel_initializer='glorot_normal',
                                             recurrent_initializer='he_uniform'))(embedding)

    drop = layers.Dropout(0.4)(blstm)

    flat = layers.Flatten()(drop)

    dense = layers.Dense(64, activation='relu')(flat)
    drop2 = layers.Dropout(0.36)(dense)

    outputs = layers.Dense(2,
                           activation='softmax')(drop2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def att_BLSTM_CNN(att_type='soft_att', embedding_model='glove', vocab_size=10, dimension=10, max_length=10,
                  embedding_matrix=None, initial_bias=None):

    inputs = layers.Input(shape=(max_length,))

    if embedding_model == 'glove' or embedding_model == 'word2vec':
        embedding = layers.Embedding(vocab_size, dimension, weights=[embedding_matrix])(inputs)

    elif embedding_model == 'bert':
        embedding = layers.Embedding(vocab_size,
                                     dimension,
                                     trainable=False,
                                     embeddings_initializer='uniform')(inputs)

    blstm = layers.Bidirectional(layers.LSTM(200,
                                             return_sequences=True,
                                             activation='tanh',
                                             recurrent_activation='sigmoid',
                                             dropout=0.44,
                                             recurrent_dropout=0.14,
                                             kernel_initializer='glorot_normal',
                                             recurrent_initializer='he_uniform'))(embedding)
    drop = layers.Dropout(0.4)(blstm)

    if att_type == 'soft_att':
        att = Attention()(drop)

    elif att_type == 'self_att':
        att = SelfAttention(size=64, num_hops=20, use_penalise=False)(drop)

    elif att_type == 'multihead_att':
        att = MultiHeadSelfAttention(dimension, num_heads=5, maxlen=max_length)(drop)

    x1 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(att)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(att)
    x2 = layers.GlobalMaxPooling1D()(x2)

    merged1 = tf.concat([x1, x2], axis=1)
    drop1 = layers.Dropout(0.5)(merged1)

    dense = layers.Dense(200, activation='relu')(drop1)
    drop2 = layers.Dropout(0.36)(dense)

    outputs = layers.Dense(2,
                          activation='softmax',
                          bias_initializer=initial_bias)(drop2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def af_att_BLSTM_CNN(att_type='soft_att', embedding_model='glove', vocab_size=10, dimension=10, max_length=10,
                     embedding_matrix=10, initial_bias=None):

    inputs = layers.Input(shape=(max_length,))

    if embedding_model == 'glove' or embedding_model == 'word2vec':
        embedding = layers.Embedding(vocab_size,
                                     dimension,
                                     weights=[embedding_matrix],
                                     embeddings_initializer='uniform',
                                     trainable=False)(inputs)

    elif embedding_model == 'bert':
        embedding = layers.Embedding(vocab_size,
                                     dimension,
                                     embeddings_initializer='uniform',
                                     input_length=max_length,
                                     trainable=False)(inputs)

    blstm = layers.Bidirectional(layers.LSTM(200,
                                             return_sequences=True,
                                             activation='tanh',
                                             recurrent_activation='sigmoid',
                                             dropout=0.44,
                                             recurrent_dropout=0.14,
                                             kernel_initializer='glorot_normal',
                                             recurrent_initializer='he_uniform',
                                             name='blstm'))(embedding)
    drop = layers.Dropout(0.4)(blstm)

    if att_type == 'soft_att':
        att = Attention()(drop)

    elif att_type == 'self_att':
        att = SelfAttention(size=64, num_hops=20, use_penalise=False)(drop)

    elif att_type == 'multihead_att':
        att = MultiHeadSelfAttention(dimension, num_heads=5, maxlen=max_length)(drop)

    inputs2 = layers.Input(shape=(20, 1))

    merged = tf.concat([att, inputs2], axis=-1)

    x1 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(merged)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(128, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='relu')(merged)
    x2 = layers.GlobalMaxPooling1D()(x2)

    merged1 = tf.concat([x1, x2], axis=1)
    drop1 = layers.Dropout(0.5)(merged1)

    dense = layers.Dense(200, activation='relu')(drop1)
    drop2 = layers.Dropout(0.36)(dense)

    dense2 = layers.Dense(2,
                          activation='softmax',
                          bias_initializer=initial_bias)(drop2)

    model = Model(inputs=[inputs, inputs2], outputs=dense2)

    return model


def get_masks(tokens, max_seq_length):

    if len(tokens) > max_seq_length:
        # Cutting down the excess length
        tokens = tokens[0: max_seq_length]
        return [1] * len(tokens)
    else:
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
      #Cutting down the excess length
        tokens = tokens[:max_seq_length]
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments
    else:
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):

    if len(tokens)>max_seq_length:
        tokens = tokens[:max_seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    else:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='val_accuracy', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
