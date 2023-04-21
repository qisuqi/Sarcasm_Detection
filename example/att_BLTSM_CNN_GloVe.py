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
    Data_Handler.pad_tweets('Harvested_Cleaned.csv', max_length, x_train, x_val, x_test)

embedding_index = Data_Handler.Load_GloVe('glove/glove.twitter.27B.200d.txt')

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Harvested_Cleaned.csv')
class_weight = {0: weight_for_0, 1: weight_for_1}

model = Models.att_BLSTM_CNN(att_type='self_att',
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

history = model.fit(padded_train_tweets, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(padded_val_tweets, y_val),
                    callbacks=[early_stopping],
                    class_weight=class_weight)

end_time = time.time()
print('Finished Training with {} seconds'.format(end_time-start_time))

Data_Handler.plot_loss(history.history['loss'],
                       history.history['val_loss'])

Data_Handler.plot_acc(history.history['accuracy'],
                      history.history['val_accuracy'])

test_result = model.evaluate(padded_test_tweets, y_test)

precision = test_result[6]
recall = test_result[7]
f1 = 2*(precision*recall)/(precision+recall)

print('Testing accuracy:', test_result[5],
      'Testing F1 Score:', f1,
      'Testing Loss:', test_result[0],
      'Testing AUC: ', test_result[8])

#writer = tf.summary.create_file_writer(logdir)
#writer.close()


tp = test_result[1].astype(int)
fp = test_result[2].astype(int)
tn = test_result[3].astype(int)
fn = test_result[4].astype(int)

cm = [[tn, fp], [fn, tp]]
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()

y_pred = model.predict(padded_test_tweets)

y_test = np.argmax(y_test, axis=-1)
y_pred = np.argmax(y_pred, axis=-1)

print(classification_report(y_test, y_pred))

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)

#tensorboard --logdir=C:\Users\Qiqi\DataScience\Datasets_Codes\TF_logs --host=127.0.0.1



