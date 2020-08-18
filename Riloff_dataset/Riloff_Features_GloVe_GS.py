import pandas as pd
import numpy as np
import math
from operator import truediv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Layer, Conv1D, MaxPool1D, Dropout, Flatten, Bidirectional, Input
from keras.layers.merge import concatenate
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import time, warnings
import Data_Handler, Models, Features

warnings.filterwarnings('ignore')

file = pd.read_csv('Riloff_tweets_cleaned.csv')


x, y = Data_Handler.load_dataset('Riloff_tweets_cleaned2.csv')

Length = []

for i in x:
    length = len(i)
    Length.append(length)

max_length = 18 #max(Length)
val_split = 0.2
dimension = 50
batch_size = 32
epochs = 20
delta = 3

PW, NW, Pos, Neg, Pos_Adj, Pos_Adv, Pos_Ver, Neg_Adj, Neg_Adv, Neg_Ver = Features.PoS_features(x)
Excl, Ques, Quos, Dots, Caps, Pos_emo, Neg_emo = Features.Punc_features(file['tweets'])


lista = (delta * PW + Pos) - (delta * NW + Neg)
listb = (delta * PW + Pos) + (delta * NW + Neg)

ratio = list(map(truediv, lista, listb))
Ratio = [0 if math.isnan(x) else x for x in ratio]
Ratio = Features.Reshape(Ratio)

aux = np.concatenate((Pos_Adj, Pos_Adv, Pos_Ver, Neg_Adj, Neg_Adv, Neg_Ver, Pos, Neg, Excl, Ques, Dots, Quos, Caps,
                           PW, NW, Ratio, Pos_emo, Neg_emo))

aux = aux.reshape((877, 18))
Aux = np.expand_dims(aux, axis=-1)

#print(pd.value_counts(file['result']))

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

encoded_tweets = t.texts_to_sequences(x)
#print(encoded_tweets)

padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')

embedding_index = Features.Load_GloVe('glove/glove.twitter.27B.50d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

x_train, y_train, x_val, y_val, x_test, y_test = Data_Handler.split_dataset(padded_tweets, y)
aux_train, aux_val, aux_test = Data_Handler.split_features(Aux)


def create_model(optimizer='adam', init='uniform'):

    inputs1 = Input(shape=(max_length,))
    embedding = Embedding(vocab_size, dimension, weights=[embedding_matrix])(inputs1)
    bltsm = Bidirectional(LSTM(4, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(embedding)
    att = Models.attention()(bltsm)

    inputs2 = Input(shape=(max_length, 1))

    merged = concatenate([att, inputs2])

    conv1 = Conv1D(64, 3, activation='relu', kernel_initializer=init)(merged)
    drop1 = Dropout(0.5)(conv1)
    conv2 = Conv1D(64, 4, activation='relu', kernel_initializer=init)(drop1)
    drop2 = Dropout(0.5)(conv2)
    # conv3 = Conv1D(64, 5, activation='relu', kernel_initializer='uniform')(drop2)
    # drop3 = Dropout(0.5)(conv3)
    maxpool1 = MaxPool1D(pool_size=2, strides=2, padding='valid')(drop2)
    flat1 = Flatten()(maxpool1)

    dense = Dense(10, activation='relu')(flat1)
    outputs = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[inputs1, inputs2], outputs=[outputs])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


start_time = time.time()

seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)

#plot_model(model, show_shapes=True, to_file='With_Features.png')

optimizers = ['adam', 'rmsprop']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [10, 20]
batchs = [5, 10]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batchs, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit([x_train, aux_train], y_train)#, validation_data=([x_val, fea_val], y_val))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





