import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
import Data_Handler, Models, Features

max_length = 20
dimension = 200
batch_size = 64
epochs = 200

#logdir = '.\TF_logs3'

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

embedding_index = Data_Handler.Load_GloVe('glove/glove.twitter.27B.200d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(0, 0, dimension)

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Harvested_Cleaned.csv')
class_weight = {0: weight_for_0, 1: weight_for_1}

model = Models.af_att_BLSTM_CNN(att_type='self_att',
                                embedding_model='glove',
                                max_length=max_length,
                                vocab_size=vocab_size,
                                dimension=dimension,
                                embedding_matrix=embedding_matrix)
                                #initial_bias=initial_bias)

print(model.summary())

#plot_model(model, show_shapes=True, to_file='af_att_BLSTM_CNN_GloVe.png')

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=0,
    patience=10,
    mode='max',
    restore_best_weights=True)

#tensorboard = TensorBoard(
 # log_dir='.\TF_logs3',
 # histogram_freq=1,
 # write_images=True
#)

#callbacks = [tensorboard]

start_time = time.time()

model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.005),
              metrics=METRICS)

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

test_result = model.evaluate([padded_test_tweets, aux_test], y_test)

#writer = tf.summary.create_file_writer(logdir)
#writer.close()

y_pred = model.predict([padded_test_tweets, aux_test])

y_test = np.argmax(y_test, axis=-1)
y_pred = np.argmax(y_pred, axis=-1)

print(classification_report(y_test, y_pred))

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)

#tensorboard --logdir=C:\Users\Qiqi\DataScience\Datasets_Codes\TF_logs --host=127.0.0.1

