import numpy as np
import pandas as pd
import time
import warnings
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import Data_Handler, Models, Features

warnings.filterwarnings('ignore')

max_length = 128
val_split  = 0.2
dimension  = 300
num_words  = 20000
batch_size = 32
epochs = 10

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

file = pd.read_csv('Riloff_tweets_cleaned2.csv')

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Riloff_tweets_cleaned2.csv')

class_weight = {0: weight_for_0, 1: weight_for_1}

x_train, x_test, x_val, y_train, y_test, y_val = Data_Handler.load_and_split_dataset('Riloff_tweets_cleaned2.csv')

t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    Data_Handler.pad_tweets('Riloff_tweets_cleaned2.csv', max_length, x_train, x_val, x_test)

word2vec = 'GoogleNews-vectors-negative300.bin'

word_vec = KeyedVectors.load_word2vec_format(word2vec, binary=True)

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    if i >= num_words:
        continue

    try:
        embedding_vector = word_vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)

start_time = time.time()

model = Models.CNN(vocab_size=vocab_size, dimension=dimension, embedding_matrix=embedding_matrix,
                   max_length=max_length, initial_bias=initial_bias)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
print(model.summary())

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


history = model.fit(padded_train_tweets, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                    validation_data=(padded_val_tweets, y_val), class_weight=class_weight)#, callbacks=[early_stopping])


test_result = model.evaluate(padded_test_tweets, y_test, batch_size=batch_size, verbose=0)
train_result = model.evaluate(padded_train_tweets, y_train, batch_size=batch_size, verbose=0)

end_time = time.time()

print('Testing Accuracy is', test_result[5]*100, 'Testing Loss is', test_result[0]*100, 'Training Accuracy is',
      train_result[5]*100, 'Training Loss is', train_result[0]*100, 'Training time is', end_time-start_time)

y_pred = model.predict(padded_test_tweets)

Data_Handler.training_curve(history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                            history.history['val_accuracy'])

Data_Handler.plot_metrics(history.history['precision'], history.history['recall'])

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred(y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

print(classification_report(y_test, y_pred))

#fpr, tpr, thres = roc_curve(y_test, y_pred)
#roc_auc = auc(fpr, tpr)
#Data_Handler.ROC_Curve(fpr, tpr, roc_auc)
