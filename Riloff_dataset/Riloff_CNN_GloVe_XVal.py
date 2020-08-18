import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPool1D
from sklearn.metrics import roc_curve, auc, classification_report
import time
import warnings

warnings.filterwarnings('ignore')

max_length = 15
val_split = 0.2

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


def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 25, weights=[embedding_matrix], input_length=max_length))
    model.add(Conv1D(5, 2, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(7, 16, padding='same', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Conv1D(11, 20, padding='same', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=10, epochs=10)

kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
results = cross_val_score(model, padded_tweets, label, cv=kfold)
pred = cross_val_predict(model, padded_tweets, label, cv=kfold)

print(results)
print(results.mean())
print(classification_report(label, pred))

fpr, tpr, _ = roc_curve(label, pred)
roc_auc = auc(fpr, tpr)

lw = 1

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
