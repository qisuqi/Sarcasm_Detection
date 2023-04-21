import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import Data_Handler, Models, Features

max_length = 20
dimension = 100
batch_size = 128
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

x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    Data_Handler.load_and_split_dataset('Harvested_Training.csv', 'Harvested_Validation.csv', 'Harvested_Testing.csv')

y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    Data_Handler.pad_tweets('Harvested_cleaned.csv', max_length, x_train, x_val, x_test)

embedding_index = Data_Handler.Load_GloVe('glove/glove.twitter.27B.100d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Harvested_Training.csv')
class_weight = {0: weight_for_0, 1: weight_for_1}

start_time = time.time()

input1 = keras.layers.Input(shape=(max_length,))
embedding = keras.layers.Embedding(vocab_size, dimension, weights=[embedding_matrix], trainable=False)(input1)
bltsm1 = keras.layers.LSTM(dimension, return_sequences=True, dropout=0.5)
bltsm2 = keras.layers.LSTM(dimension, return_sequences=True, go_backwards=True, dropout=0.5)
bidirectional = keras.layers.Bidirectional(bltsm1, backward_layer=bltsm2)(embedding)
att = Models.MultiHeadSelfAttention(200, num_heads=4, maxlen=max_length)(bidirectional)

input2 = keras.layers.Input(shape=(20, 1))

merged = tf.concat([att, input2], axis=-1)

flat1 = keras.layers.Flatten()(merged)
outputs = keras.layers.Dense(2, activation='softmax')(flat1)

model = keras.Model(inputs=[input1, input2], outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)

print(model.summary())

#plot_model(model, show_shapes=True, to_file='Kumar.png')

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

history = model.fit([padded_train_tweets, aux_train], y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=([padded_val_tweets, aux_val], y_val),
                    callbacks=[early_stopping])

end_time = time.time()
print('Finished Training with {} seconds'.format(end_time-start_time))

Data_Handler.plot_loss(history.history['loss'],
                       history.history['val_loss'])

Data_Handler.plot_acc(history.history['accuracy'],
                      history.history['val_accuracy'])

model.save('Kumar_Harvested')





