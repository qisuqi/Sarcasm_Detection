import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Dropout, Flatten, ReLU
import warnings
import Data_Handler, Models, Features

warnings.filterwarnings('ignore')

x, y = Data_Handler.load_dataset('Riloff_tweets_cleaned2.csv')

Length = []

for i in x:
    length = len(i)
    Length.append(length)

max_length = 18
val_split  = 0.2
dimension  = 25

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

encoded_tweets = t.texts_to_sequences(x)
padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')

embedding_index = Features.Load_GloVe('glove/glove.twitter.27B.50d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

x_train, y_train, x_val, y_val, x_test, y_test = Data_Handler.split_dataset(padded_tweets, y)


def create_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Embedding(vocab_size, dimension, weights=[embedding_matrix], input_length=max_length))
    model.add(Conv1D(128, 3, kernel_initializer=init))
    model.add(ReLU())
    model.add(Conv1D(128, 4, padding='same', kernel_initializer=init))
    model.add(ReLU())
    model.add(Conv1D(128, 5, padding='same', kernel_initializer=init))
    model.add(ReLU())
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers = ['adam', 'rmsprop']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [10, 20, 50]
batchs = [10, 32, 64]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batchs, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train, validation_data=(x_val, y_val))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


