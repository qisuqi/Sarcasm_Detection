import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPool1D, ReLU, Dropout, BatchNormalization, Activation
from sklearn.metrics import roc_curve, auc, classification_report
import time
import warnings

warnings.filterwarnings('ignore')


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

Length = []

for i in x:
    length = len(i)
    Length.append(length)

max_length = max(Length)
val_split  = 0.2
dimension  = 25
num_words  = 20000

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

embedding_matrix = np.zeros((vocab_size, dimension))

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


def create_model(size1=128, size2=128, size3=128, num=3):
    model = Sequential()
    model.add(Embedding(vocab_size, 25, weights=[embedding_matrix], input_length=max_length))
    model.add(Conv1D(size1, num, activation='relu', kernel_initializer='uniform'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(size2, num, padding='same', activation='relu', kernel_initializer='uniform'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(size3, num, padding='same', activation='relu', kernel_initializer='uniform'))
    #model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)

size1 = [64, 58]
size2 = [58, 32]
size3 = [32, 16]

num = [3, 4, 5]

param_grid = dict(size1=size1, size2=size2, size3=size3, num=num)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train, validation_data=(x_val, y_val))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


