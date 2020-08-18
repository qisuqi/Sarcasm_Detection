from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPool1D, Dropout, Flatten, Bidirectional, ReLU, Input
from keras.layers.merge import concatenate
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Permute, Activation


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
        #embedding_matrix_flattened = Flatten()(embedding_matrix)

        if self.use_penalise:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)
            product = tf.matmul(attention_weights, attention_weights_transposed)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0], ))
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product-identity)))
            self.add_loss(self.penalty_coeff * frobenius_norm)

        return embedding_matrix


def att_BLTSM_CNN(vocab_size=10, dimension=10, embedding_matrix=10, max_length=18):
    model = Sequential()
    model.add(Embedding(vocab_size, dimension, weights=[embedding_matrix], input_length=max_length))
    model.add(Attention())
    model.add(Bidirectional(LSTM(9, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)))
    model.add(Conv1D(128, 3, kernel_initializer='normal'))
    model.add(ReLU())
    model.add(Conv1D(128, 4, padding='same', kernel_initializer='normal'))
    model.add(ReLU())
    model.add(Conv1D(128, 5, padding='same', kernel_initializer='normal'))
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


class BERT(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim=128, cnn_filters=100, dnn_units=512,
                 ltsm_unit=64, dropout_rate=0.1, training=False, name='sarcasm_model'):

        super(BERT, self).__init__(name=name)

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.bltsm = layers.Bidirectional(layers.LSTM(ltsm_unit, dropout=dropout_rate, return_sequences=True))
        self.attention = SelfAttention(size=128, num_hops=10, use_penalise=False)
        self.cnn1 = layers.Conv1D(filters=cnn_filters, kernel_size=3, padding='valid', activation='relu')
        self.cnn2 = layers.Conv1D(filters=cnn_filters, kernel_size=4, padding='valid', activation='relu')
        self.cnn3 = layers.Conv1D(filters=cnn_filters, kernel_size=5, padding='valid', activation='relu')
        self.pool = layers.GlobalMaxPooling1D()
        self.dense1 = layers.Dense(units=dnn_units, activation='relu')
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.dense2 = layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.bltsm(l)
        l_1 = self.attention(l_1)
        l_1 = self.cnn1(l_1)
        l_1 = self.pool(l_1)
        l_2 = self.cnn2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn3(l)
        l_3 = self.pool(l_3)

        concat = tf.concat([l_1, l_2, l_3], axis=-1)
        concat = self.dense1(concat)
        concat = self.dropout(concat, training)
        output = self.dense2(concat)

        return output


def satt_BLTSM_CNN(max_length=10, vocab_size=10, dimension=10, embedding_matrix=10):

    inputs = layers.Input(shape=(max_length, ))
    embedding = layers.Embedding(vocab_size, dimension, weights=[embedding_matrix])(inputs)
    bltsm = layers.Bidirectional(layers.LSTM(9, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(embedding)
    att = SelfAttention(size=128, num_hops=10, use_penalise=False)(bltsm)

    conv1 = layers.Conv1D(128, 3, kernel_initializer='normal')(att)
    relu1 = layers.ReLU()(conv1)
    conv2 = layers.Conv1D(128, 4, padding='same', kernel_initializer='normal')(relu1)
    relu2 = layers.ReLU()(conv2)
    conv3 = layers.Conv1D(128, 5, padding='same', kernel_initializer='normal')(relu2)
    relu3 = layers.ReLU()(conv3)
    maxpool1 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')(relu3)
    drop = layers.Dropout(0.5)(maxpool1)
    flat1 = layers.Flatten()(drop)

    outputs = layers.Dense(1, activation='sigmoid')(flat1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def CNN(vocab_size=10, dimension=10, embedding_matrix=10, max_length=10):
    model = Sequential()
    model.add(Embedding(vocab_size, dimension, weights=[embedding_matrix], input_length=max_length))
    model.add(Conv1D(128, 3, kernel_initializer='uniform'))
    model.add(ReLU())
    model.add(Conv1D(128, 4, padding='same', kernel_initializer='uniform'))
    model.add(ReLU())
    model.add(Conv1D(128, 5, padding='same', kernel_initializer='uniform'))
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def af_att_BLTSM_CNN(max_length=10, vocab_size=10, dimension=10, embedding_matrix=10):

    inputs1 = Input(shape=(max_length,))
    embedding = Embedding(vocab_size, dimension, weights=[embedding_matrix])(inputs1)
    bltsm = Bidirectional(LSTM(9, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(embedding)
    att = Attention()(bltsm)

    inputs2 = Input(shape=(max_length, 1))

    merged = concatenate([att, inputs2])

    conv1 = Conv1D(128, 3, kernel_initializer='normal')(merged)
    relu1 = ReLU()(conv1)
    conv2 = Conv1D(128, 4, padding='same', kernel_initializer='normal')(relu1)
    relu2 = ReLU()(conv2)
    conv3 = Conv1D(128, 5, padding='same', kernel_initializer='normal')(relu2)
    relu3 = ReLU()(conv3)
    maxpool1 = MaxPool1D(pool_size=2, strides=2, padding='valid')(relu3)
    drop = Dropout(0.5)(maxpool1)
    flat1 = Flatten()(drop)

    outputs = Dense(1, activation='sigmoid')(flat1)

    model = Model(inputs=[inputs1, inputs2], outputs=[outputs])

    return model