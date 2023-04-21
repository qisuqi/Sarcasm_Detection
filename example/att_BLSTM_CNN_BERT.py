import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bert.tokenization import bert_tokenization
import tensorflow.keras as keras
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from tensorflow.keras.utils import plot_model
from sklearn import metrics
import Data_Handler, Models, Features

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

BertTokenizer = bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2",
                            trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

vocab_length = len(tokenizer.vocab)
embedding_dim = 200
nb_epochs = 200
batch_size = 64
max_size = 20


def tokenise_tweets(tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))


tokenised_train_tweets = [tokenise_tweets(tweet) for tweet in x_train]
padded_train_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_train_tweets, maxlen=max_size, padding='post')

tokenised_val_tweets = [tokenise_tweets(tweet) for tweet in x_val]
padded_val_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_val_tweets, maxlen=max_size, padding='post')

tokenised_test_tweets = [tokenise_tweets(tweet) for tweet in x_test]
padded_test_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_test_tweets, maxlen=max_size, padding='post')

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('Harvested_Cleaned.csv')
class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=0,
    patience=10,
    mode='max',
    restore_best_weights=True)

call_backs = Models.TerminateOnBaseline(
        monitor='val_auc',
        baseline=0.98)

model = Models.att_BLSTM_CNN(att_type='self_att',
                                embedding_model='bert',
                                vocab_size=vocab_length,
                                dimension=embedding_dim,
                                max_length=max_size)
                                #initial_bias=initial_bias)

print(model.summary())

#plot_model(model, show_shapes=True, to_file='af-satt-BLSTM-CNN-BERT.png')

start_time = time.time()

model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=0.005),
              metrics=METRICS)

history = model.fit(padded_train_tweets, y_train,
                    validation_data=(padded_val_tweets, y_val),
                    batch_size=batch_size,
                    epochs=nb_epochs,
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

y_pred = np.argmax(y_pred, axis=-1)
y_test = np.argmax(y_test, axis=1)

print(y_pred[105], y_test[105])

print(metrics.classification_report(y_test, y_pred))

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)
