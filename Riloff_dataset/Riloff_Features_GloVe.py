import pandas as pd
import numpy as np
from operator import truediv
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc, classification_report
import time, warnings
import Data_Handler, Features, Models

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
aux_train, aux_val, aux_test = Data_Handler.split_features(Aux)

start_time = time.time()

model = Models.af_att_BLTSM_CNN(max_length=max_length, vocab_size=vocab_size, dimension=dimension,
                                embedding_matrix=embedding_matrix)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

#plot_model(model, show_shapes=True, to_file='With_Features.png')

#callbacks = [EarlyStopping(monitor='loss')]

history = model.fit([x_train, aux_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                    validation_data=([x_val, aux_val], y_val))#, callbacks=callbacks)

test_loss, test_accuracy = model.evaluate([x_test, aux_test], y_test, batch_size=batch_size, verbose=0)
train_loss, train_accuracy = model.evaluate([x_train, aux_train], y_train, batch_size=batch_size, verbose=0)

end_time = time.time()

print('Testing Accuracy is', test_accuracy*100, 'Testing Loss is', test_loss*100, 'Training Accuracy is',
      train_accuracy*100, 'Training Loss is', train_loss*100, 'Training time is', end_time-start_time)

y_pred = model.predict([x_test, aux_test])

fpr, tpr, thres = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

print(classification_report(y_test, y_pred))

Data_Handler.Training_Curve(history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                            history.history['val_accuracy'])

Data_Handler.ROC_Curve(fpr, tpr, roc_auc)



