import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
from gensim.models import KeyedVectors
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
import Data_Handler, Models, Features

max_length = 20
dimension = 300
batch_size = 64
num_words  = 20000
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

google = 'GoogleNews-vectors-negative300.bin.gz'

word_vec = KeyedVectors.load_word2vec_format(google, binary=True)

embedding_matrix = np.zeros((vocab_size, dimension))

for word, i in t.word_index.items():
    if i >= num_words:
        continue

    try:
        embedding_vector = word_vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dimension)


model = Models.att_BLSTM_CNN(att_type='self_att',
                                embedding_model='word2vec',
                                max_length=max_length,
                                vocab_size=vocab_size,
                                dimension=dimension,
                                embedding_matrix=embedding_matrix)

print(model.summary())

#plot_model(model, show_shapes=True, to_file='af-satt-BLTSM-CNN-word2vec.png')

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

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

end_time = time.time()

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
#Data_Handler.ROC_Curve(fpr, tpr, roc_auc)