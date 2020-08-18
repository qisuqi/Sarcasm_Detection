import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, Layer, Conv1D, MaxPool1D, Dropout, Flatten, Bidirectional, ReLU, Attention, GlobalMaxPooling1D, Concatenate
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
import keras.backend as K
import keras
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

max_length = 18
val_split = 0.2
dimension = 50
batch_size = 32
epochs = 20

#print(pd.value_counts(file['result']))

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

encoded_tweets = t.texts_to_sequences(x)
#print(encoded_tweets)

padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')
#print(padded_tweets)

embedding_index = dict()

f = open('glove/glove.twitter.27B.50d.txt')
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
#np.random.shuffle(indices)
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

print('Training set class distribution', pd.value_counts(y_train))
print('Validation set class distribution', pd.value_counts(y_val))
print('Testing set class distribution', pd.value_counts(y_test))

start_time = time.time()

query_input = Input(shape=(max_length, ), dtype='int32')
value_input = Input(shape=(1, ), dtype='int32')

embedding = Embedding(vocab_size, dimension, weights=[embedding_matrix])

query_embedding = embedding(query_input)
value_embedding = embedding(value_input)

bltsm = Bidirectional(LSTM(9, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))

query_seq_enc = bltsm(query_embedding)
value_seq_enc = bltsm(value_embedding)

query_value_attention = Attention(dropout=0.2)([query_seq_enc, value_seq_enc])

#query_enc = GlobalAveragePooling1D()(query_seq_enc)
#query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)

input_layer = Concatenate()([query_seq_enc, query_value_attention])

conv1 = Conv1D(128, 3)(input_layer)
relu1 = ReLU()(conv1)
drop1 = Dropout(0.5)(relu1)
conv2 = Conv1D(128, 4)(relu1)
relu2 = ReLU()(conv2)
drop2 = Dropout(0.5)(relu2)
conv3 = Conv1D(128, 5)(relu2)
relu3 = ReLU()(conv3)
drop3 = Dropout(0.5)(relu3)
#maxpool1 = MaxPool1D(pool_size=2, strides=2, padding='valid')(drop3)
#flat = Flatten()(drop)

maxpool = GlobalMaxPooling1D()(drop3)
dense = Dense(10, activation='softmax')(maxpool)
outputs = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[query_input, value_input], outputs=outputs)

opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

plot_model(model, show_shapes=True, to_file='att-BLTSM-CNN1.png')

callbacks = [EarlyStopping(monitor='val_loss')]

history = model.fit([x_train, y_train], y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                    validation_data=([x_val, x_val], y_val), callbacks=callbacks)

test_loss, test_accuracy = model.evaluate([x_test, y_test], y_test, batch_size=batch_size, verbose=0)
train_loss, train_accuracy = model.evaluate([x_train, y_train], y_train, batch_size=batch_size, verbose=0)

end_time = time.time()

print('Testing Accuracy is', test_accuracy*100, 'Testing Loss is', test_loss*100, 'Training Accuracy is',
      train_accuracy*100, 'Training Loss is', train_loss*100, 'Training time is', end_time-start_time)

y_pred = model.predict([x_test, y_test])
print(y_pred)

fpr, tpr, thres = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

#print(y_test, y_pred)

print(classification_report(y_test, y_pred))

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Evaluation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.show()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

