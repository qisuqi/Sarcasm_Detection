import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import Data_Handler, Models, Features

max_length = 20
dimension = 200
batch_size = 32
epochs = 100

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

s = ["you guys im so proud we keep beating our record every day lets keep this up",
     "tuesday mood wait until the weekend"]
y_train = [[1], [0]]
y_train = keras.utils.to_categorical(y_train)

tokeniser = Tokenizer()
tokeniser.fit_on_texts(s)
vocab_size = len(tokeniser.word_index) + 1

encoded_train_tweets = tokeniser.texts_to_sequences(s)
padded_train_tweets = pad_sequences(encoded_train_tweets, maxlen=max_length, padding='post')
#padded_train_tweets = tf.convert_to_tensor(padded_train_tweets, dtype=np.float32)
#padded_train_tweets = tf.expand_dims(padded_train_tweets, axis=-1)

embedding_index = Data_Handler.Load_GloVe('glove/glove.twitter.27B.200d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in tokeniser.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input = layers.Input(shape=(max_length, ))
embedding = layers.Embedding(vocab_size, dimension, weights=[embedding_matrix])(input)
blstm = layers.Bidirectional(layers.LSTM(200, return_sequences=True))(embedding)
att, weights = Models.SelfAttention(size=64, num_hops=20, use_penalise=False)(blstm)
flat = layers.Flatten()(att)
output = layers.Dense(2, activation='softmax')(flat)
model = keras.Model(inputs=input, outputs=output)

model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.005),
              metrics=METRICS)

history = model.fit(padded_train_tweets, y_train)

outputs = []
for layer in model.layers:
    func = K.function([model.input], [layer.output])
    outputs.append(func([padded_train_tweets, 1]))

att = outputs[3]
print(att[0][0][0].shape)
print(att[0][1][0].shape)

sns.heatmap(att[0][0][0])
#plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
 #          ['you', 'guys', 'im', 'so', 'proud', 'we', 'keep', 'beating', 'our', 'record',
  #          'every', 'day', 'lets', 'this', 'up', 'pad', 'pad', 'pad', 'pad', 'pad'])
#plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
 #          ['you', 'guys', 'im', 'so', 'proud', 'we', 'keep', 'beating', 'our', 'record',
  #          'every', 'day', 'lets', 'this', 'up', 'pad', 'pad', 'pad', 'pad', 'pad'])
#plt.yticks(rotation=360)
#plt.xticks(rotation=70)
plt.title('Attention Weights Heatmap')
plt.show()

sns.heatmap(att[0][1][0])
plt.show()

