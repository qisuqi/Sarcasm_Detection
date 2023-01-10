import numpy as np
import time
from bert.tokenization import bert_tokenization
import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import shuffle
import Data_Handler, Models, Features

#logdir = '.\TF_logs'

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
    Data_Handler.load_and_split_dataset('SemEval_Training.csv', 'SemEval_Testing.csv')

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('SemEval_cleaned.csv')

class_weight = {0: weight_for_0, 1: weight_for_1}
print(class_weight)

BertTokenizer = bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2",
                            trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

vocab_length = len(tokenizer.vocab)
embedding_dim = 200
nb_epochs = 100
batch_size = 64
num_runs = 5
max_size = 20

def tokenise_tweets(tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))


tokenised_train_tweets = [tokenise_tweets(tweet) for tweet in x_train]
padded_train_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_train_tweets, maxlen=max_size, padding='post')

tokenised_val_tweets = [tokenise_tweets(tweet) for tweet in x_val]
padded_val_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_val_tweets, maxlen=max_size, padding='post')

tokenised_test_tweets = [tokenise_tweets(tweet) for tweet in x_test]
padded_test_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_test_tweets, maxlen=max_size, padding='post')

Y_pred = []

for i in range(num_runs):

    inputs = layers.Input(shape=(max_size,))
    embedding = layers.Embedding(vocab_length,
                                 embedding_dim,
                                 trainable=True,
                                 embeddings_initializer='uniform')(inputs)
    drop = layers.Dropout(0.43)(embedding)
    blstm = layers.Bidirectional(layers.LSTM(100,
                                             return_sequences=True,
                                             activation='tanh',
                                             recurrent_activation='sigmoid',
                                             dropout=0.44,
                                             recurrent_dropout=0.14,
                                             kernel_initializer='glorot_normal',
                                             recurrent_initializer='he_uniform',
                                             kernel_regularizer='l2',
                                             recurrent_regularizer='l2'))(drop)
    att = Models.SelfAttention(size=64, num_hops=20, use_penalise=False)(blstm)

    inputs2 = layers.Input(shape=(max_size, 1))
    merged = tensorflow.concat([att, inputs2], axis=-1)

    x1 = layers.Conv1D(100, 3,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(100, 4,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x3 = layers.Conv1D(100, 5,
                       padding='valid',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x3 = layers.GlobalMaxPooling1D()(x3)

    merged1 = tensorflow.concat([x1, x2, x3], axis=1)
    batch_norm = layers.BatchNormalization()(merged1)
    dense = layers.Dense(352, activation='relu')(batch_norm)
    drop1 = layers.Dropout(0.36)(dense)

    dense2 = layers.Dense(1,
                          activation='sigmoid',
                          bias_initializer=initial_bias,
                          kernel_regularizer='l2')(drop1)

    model = keras.Model(inputs=[inputs, inputs2], outputs=dense2)

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                  metrics=METRICS)

    #plot_model(model, show_shapes=True, to_file='af-satt-BLSTM-CNN-BERT.png')

    early_stopping = EarlyStopping(
                    monitor='val_auc',
                    verbose=0,
                    patience=10,
                    mode='max',
                    restore_best_weights=True)

    #tensorboard = TensorBoard(
      #log_dir='.\TF_logs',
      #histogram_freq=1,
      #write_images=True
    #)

    #callbacks = [tensorboard, EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)]

    history = model.fit([padded_train_tweets, aux_train], y_train,
                        batch_size=batch_size,
                        epochs=nb_epochs,
                        validation_data=([padded_val_tweets, aux_val], y_val),
                        callbacks=[early_stopping],
                        class_weight=class_weight)
                        #steps_per_epoch=np.ceil(2453//batch_size))

    train_result = model.evaluate([padded_train_tweets, aux_train], y_train, batch_size=batch_size)
    test_result = model.evaluate([padded_test_tweets, aux_test], y_test, batch_size=batch_size)

    #writer = tf.summary.create_file_writer(logdir)
    #writer.close()

    # print(model.summary())

    y_pred = model.predict([padded_test_tweets, aux_test], batch_size=batch_size)
    y_pred = y_pred.round()

    Y_pred.append(y_pred)

Y_pred = np.array(Y_pred).reshape((len(y_test), num_runs))
import pandas as pd
pd.DataFrame(Y_pred).to_csv('pred.csv')
pd.DataFrame(y_test).to_csv('test.csv')


F1_0 = []
F1_1 = []
F1_weight = []
Precision_0 = []
Precision_1 = []
Precision_weight = []
Recall_0 = []
Recall_1 = []
Recall_weight = []
Accuracy = []
ROC_AUC = []
FPR = []
TPR = []
CM = []

for i in range(num_runs):
    f1_0 = f1_score(y_test, Y_pred[:, i], pos_label=1)
    f1_1 = f1_score(y_test, Y_pred[:, i], pos_label=0)
    f1_weight = f1_score(y_test, Y_pred[:, i], average='weighted')
    precision_0 = precision_score(y_test, Y_pred[:, i], pos_label=0)
    precision_1 = precision_score(y_test, Y_pred[:, i], pos_label=1)
    precision_weight = precision_score(y_test, Y_pred[:, i], average='weighted')
    recall_0 = recall_score(y_test, Y_pred[:, i], pos_label=0)
    recall_1 = recall_score(y_test, Y_pred[:, i], pos_label=1)
    recall_weight = recall_score(y_test, Y_pred[:, i], average='weighted')
    accuracy = accuracy_score(y_test, Y_pred[:, i])
    fpr, tpr, _ = roc_curve(y_test, Y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, Y_pred[:, i])

    F1_0.append(f1_0)
    F1_1.append(f1_1)
    F1_weight.append(f1_weight)
    Precision_0.append(precision_0)
    Precision_1.append(precision_1)
    Precision_weight.append(precision_1)
    Recall_0.append(recall_0)
    Recall_1.append(recall_1)
    Recall_weight.append(recall_weight)
    Accuracy.append(accuracy)
    FPR.append(fpr)
    TPR.append(tpr)
    ROC_AUC.append(roc_auc)
    CM.append(cm)

print(np.average(F1_0), np.average(F1_1), np.average(F1_weight))
print(np.average(Precision_0), np.average(Precision_1), np.average(Precision_weight))
print(np.average(Recall_0), np.average(Recall_1), np.average(Recall_weight))
print(np.average(Accuracy))
print(np.average(FPR), np.average(TPR))
print(np.average(ROC_AUC))

result = np.array([F1_weight, Precision_weight, Recall_weight, Accuracy]).T
plt.boxplot(result)
plt.ylabel('Score')
plt.xticks([1, 2, 3, 4], ['F1 Score', 'Precision', 'Recall', 'Accuracy'])
plt.title('KPIs Distribution')
plt.show()

plt.plot(FPR[0], TPR[0], color='darkorange', label='ROC curve1 (area = %0.2f)' % ROC_AUC[0])
plt.plot(FPR[1], TPR[1], color='red', label='ROC curve2 (area = %0.2f)' % ROC_AUC[1])
plt.plot(FPR[2], TPR[2], color='tomato', label='ROC curve3 (area = %0.2f)' % ROC_AUC[2])
plt.plot(FPR[3], TPR[3], color='firebrick', label='ROC curve4 (area = %0.2f)' % ROC_AUC[3])
plt.plot(FPR[4], TPR[4], color='indianred', label='ROC curve5 (area = %0.2f)' % ROC_AUC[4])
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

CM = np.array(CM)

fp = [CM[0][0][1], CM[1][0][1], CM[2][0][1], CM[3][0][1], CM[4][0][1]]
tp = [CM[0][0][0], CM[1][0][0], CM[2][0][0], CM[3][0][0], CM[4][0][0]]
fn = [CM[0][1][0], CM[1][1][0], CM[2][1][0], CM[3][1][0], CM[4][1][0]]
tn = [CM[0][1][1], CM[1][1][1], CM[2][1][1], CM[3][1][1], CM[4][1][1]]

fp = np.average(fp).astype(int)
tp = np.average(tp).astype(int)
fn = np.average(fn).astype(int)
tn = np.average(tn).astype(int)

cm_test = [[tp, fp], [fn, tn]]
sns.heatmap(cm_test, annot=True, fmt="d")
plt.title('Testing Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()

#sns.regplot(y_test, Y_pred[0], color='slateblue', label='Run1')
#sns.regplot(y_test, Y_pred[1], color='rebeccapurple', label='Run2')
#sns.regplot(y_test, Y_pred[2], color='darkviolet', label='Run3')
#sns.regplot(y_test, Y_pred[3], color='darkorchid', label='Run4')
#sns.regplot(y_test, Y_pred[4], color='purple', label='Run5')
#plt.ylabel('Predicted Labels')
#plt.xlabel('True Labels')
#plt.title('Regression Line for the Predicted and True Labels')
#plt.show()


#print('Testing Accuracy is', test_result[5]*100, 'Testing Loss is', test_result[0]*100, 'Training Accuracy is',
 #     train_result[5]*100, 'Training Loss is', train_result[0]*100, 'Training time is', end_time-start_time)

#Data_Handler.training_curve(history.history['loss'], history.history['accuracy'])

#Data_Handler.plot_metrics(history.history['precision'], history.history['recall'])

#Data_Handler.plot_cm(y_test, y_pred)

#Data_Handler.plot_roc(y_test, y_pred)

#Data_Handler.plot_ypred(y_pred)

#Data_Handler.plot_ypred_ytest(y_test, y_pred)

#y_pred[y_pred <= 0.5] = 0.
#y_pred[y_pred > 0.5] = 1.

#print(classification_report(y_test, y_pred))


#| -Score: 0.6368077993392944
#| -Best
#step: 0
#> Hyperparameters:
#| -BLSTM: 100
#| -Dense: 32
#| -FilterNum: 256
#| -FilterSize: 3
#| -conv1_init: he_uniform
#| -conv_init: he_uniform
#| -dr: 0.1819388102569894
#| -emb_init: he_uniform
#| -lr: 0.0008068132443472298
#| -lstm1_act: tanh
#| -lstm1_init: normal
#| -lstm_act: sigmoid
#| -lstm_init: he_uniform
#| -padding: same
#| -padding1: same

#tensorboard --logdir=C:\Users\Qiqi\DataScience\Datasets_Codes\TF_logs --host=127.0.0.1
#Average testing accuracy is:  62.00782299041748 Average training accuracy is:  83.11455249786377 Average testing F1 score is:  0.5918574507273923 Average training F1 score is:  0.8222223718167208