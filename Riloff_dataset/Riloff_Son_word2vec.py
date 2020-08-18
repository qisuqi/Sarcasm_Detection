import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import simple_preprocess
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import roc_curve, auc, classification_report
import time, warnings
import Data_Handler, Models, Features

warnings.filterwarnings('ignore')

x, y = Data_Handler.load_dataset('Riloff_tweets_cleaned2.csv')

Length = []

for i in x:
    length = len(i)
    Length.append(length)

max_length = 18 #max(Length)
val_split  = 0.2
dimension  = 300
num_words  = 20000
batch_size = 32
epochs = 10

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

encoded_tweets = t.texts_to_sequences(x)
padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')

file = 'GoogleNews-vectors-negative300.bin'

word_vec = KeyedVectors.load_word2vec_format(file, binary=True)

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    if i >= num_words:
        continue

    try:
        embedding_vector = word_vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)

x_train, y_train, x_val, y_val, x_test, y_test = Data_Handler.split_dataset(padded_tweets, y)

start_time = time.time()

model = Models.satt_BLTSM_CNN(vocab_size=vocab_size, dimension=dimension, embedding_matrix=embedding_matrix,
                              max_length=max_length)

#opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

#plot_model(model, show_shapes=True, to_file='att-BLTSM-CNN.png')

#callbacks = [EarlyStopping(monitor='val_loss')]

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
                    #callbacks=callbacks)

test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
train_loss, train_accuracy = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)

end_time = time.time()

print('Testing Accuracy is', test_accuracy*100, 'Testing Loss is', test_loss*100, 'Training Accuracy is',
      train_accuracy*100, 'Training Loss is', train_loss*100, 'Training time is', end_time-start_time)

y_pred = model.predict(x_test)

fpr, tpr, thres = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

#print(y_test, y_pred)

print(classification_report(y_test, y_pred))

Data_Handler.Training_Curve(history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                      history.history['val_accuracy'])

Data_Handler.ROC_Curve(fpr, tpr, roc_auc)

