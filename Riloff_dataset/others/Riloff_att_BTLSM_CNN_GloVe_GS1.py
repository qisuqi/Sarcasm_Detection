import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPool1D, Bidirectional, LSTM, Layer, ReLU
from sklearn.metrics import roc_curve, auc, classification_report
import time
import warnings

warnings.filterwarnings('ignore')

max_length = 18
val_split = 0.2
dimension = 25

def load_dataset(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    x = dataset[:, -1]
    y = dataset[:, 2]
    x = x.astype(str)
    y = y.reshape((len(y), 1))
    return x, y

def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return y_enc

x, y = load_dataset('Riloff_tweets_cleaned2.csv')

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

encoded_tweets = t.texts_to_sequences(x)
#print(encoded_tweets)

padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')
#print(padded_tweets)

embedding_index = dict()

f = open('glove/glove.twitter.27B.25d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype=float)
    embedding_index[word] = coefs
f.close()

print('Loaded %s word vectors' % len(embedding_index))

embedding_matrix = np.zeros((vocab_size, 25))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

indices = np.arange(padded_tweets.shape[0])
np.random.shuffle(indices)
padded_tweets = padded_tweets[indices]
label1 = y[indices]
label = prepare_targets(label1)
num_val = int(val_split * padded_tweets.shape[0])

x_train = padded_tweets[:-num_val]
y_train = label[:-num_val]
x_val = x_train[-num_val:]
y_val = y_train[-num_val:]
x_test = padded_tweets[-num_val:]
y_test = label[-num_val:]

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], input_shape[1], 1), initializer='zeros')

        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


def create_model(dropout=0.3, recurrent_dropout=0.2): #, size3=7, num3=20
    model = Sequential()
    model.add(Embedding(vocab_size, dimension, weights=[embedding_matrix], input_length=max_length))
    model.add(Bidirectional(LSTM(9, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(attention())
    model.add(Conv1D(128, 3, kernel_initializer='glorot_uniform'))
    model.add(ReLU())
    model.add(Conv1D(128, 4, padding='same', kernel_initializer='glorot_uniform'))
    model.add(ReLU())
    model.add(Conv1D(128, 5, padding='same', kernel_initializer='glorot_uniform'))
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)

dropout = [0.1, 0,3, 0.5]
recurrent_dropout = [0.1, 0.2, 0.4]

param_grid = dict(dropout=dropout, recurrent_dropout=recurrent_dropout) #, size3=size3, , num3=num3)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train, validation_data=(x_val, y_val))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


