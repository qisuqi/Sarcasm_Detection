import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import Data_Handler, Models, Features

max_length = 20
dimension = 200
batch_size = 10
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

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Harvested_Cleaned.csv')

class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    Data_Handler.load_and_split_dataset('Harvested_Training.csv', 'Harvested_Validation.csv', 'Harvested_Testing.csv')

y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    Data_Handler.pad_tweets('Harvested_Cleaned.csv', max_length, x_train, x_val, x_test)

embedding_index = Data_Handler.Load_GloVe('glove/glove.twitter.27B.200d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)


init = keras.initializers.VarianceScaling(scale=0.01, distribution='truncated_normal')
bias = keras.initializers.Constant(0)
blstm_bias = keras.initializers.Constant(1)

inputs = keras.layers.Input(shape=(max_length,))
embedding = keras.layers.Embedding(vocab_size,
                                   dimension,
                                   weights=[embedding_matrix],
                                   trainable=True)(inputs)
drop = keras.layers.Dropout(0.5)(embedding)
bltsm1 = keras.layers.LSTM(500,
                           return_sequences=True,
                           dropout=0.2,
                           kernel_initializer=init,
                           bias_initializer=blstm_bias)
bltsm2 = keras.layers.LSTM(500,
                           return_sequences=True,
                           go_backwards=True,
                           dropout=0.2,
                           kernel_initializer=init,
                           bias_initializer=blstm_bias)
bltsm = keras.layers.Bidirectional(bltsm1,
                                   backward_layer=bltsm2)(drop)

att = Models.Attention()(bltsm)

inputs2 = keras.layers.Input(shape=(20, 1))

merged = tf.concat([att, inputs2], axis=-1)

x1 = keras.layers.Conv1D(100, 3,
                            padding='valid',
                            kernel_initializer=init,
                            bias_initializer=bias)(merged)
x1 = keras.layers.ReLU()(x1)
x2 = keras.layers.Conv1D(100, 3,
                            padding='valid',
                            kernel_initializer=init,
                            bias_initializer=bias)(merged)
x2 = keras.layers.ReLU()(x2)
x3 = keras.layers.Conv1D(100, 3,
                            padding='valid',
                            kernel_initializer=init,
                            bias_initializer=bias)(merged)
x3 = keras.layers.ReLU()(x3)
drop1 = keras.layers.Dropout(0.4)(x3)
maxpool = keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid')(drop1)
flat = keras.layers.Flatten()(maxpool)

outputs = keras.layers.Dense(2,
                             activation='softmax',
                             kernel_initializer=init,
                             bias_initializer=bias)(flat)

model = keras.Model(inputs=[inputs, inputs2], outputs=[outputs])

print(model.summary())

#plot_model(model, show_shapes=True, to_file='Son.png')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.2),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)

early_stopping = EarlyStopping(
        monitor='val_auc',
        verbose=0,
        patience=10,
        mode='max',
        restore_best_weights=True)

call_backs = Models.TerminateOnBaseline(
        monitor='val_auc',
        baseline=0.98)

start_time = time.time()

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

#model.save('Son_Harvested')

