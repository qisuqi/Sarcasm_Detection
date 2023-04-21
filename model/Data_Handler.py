import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_and_split_dataset(train, val, test):
    train_file = pd.read_csv(train)
    val_file = pd.read_csv(val)
    test_file = pd.read_csv(test)

    train_data = train_file.values
    val_data = val_file.values
    test_data = test_file.values

    x_train = train_data[:, 1].astype(str)
    aux_train = train_data[:, 2:]
    aux_train = np.array(aux_train).reshape((len(aux_train), 20))
    aux_train = tf.convert_to_tensor(aux_train, dtype=np.float32)
    aux_train = tf.expand_dims(aux_train, axis=-1)

    x_val = val_data[:, 1].astype(str)
    aux_val = val_data[:, 2:]
    aux_val = np.array(aux_val).reshape((len(aux_val), 20))
    aux_val = tf.convert_to_tensor(aux_val, dtype=np.float32)
    aux_val = tf.expand_dims(aux_val, axis=-1)

    x_test = test_data[:, 1].astype(str)
    aux_test = test_data[:, 2:]
    aux_test = np.array(aux_test).reshape((len(aux_test), 20))
    aux_test = tf.convert_to_tensor(aux_test, dtype=np.float32)
    aux_test = tf.expand_dims(aux_test, axis=-1)

    y_train = train_data[:, 0]
    y_train = prepare_targets(y_train)
    y_train = y_train.reshape((len(y_train), 1))
    y_train = tf.convert_to_tensor(y_train, dtype=np.float32)

    y_val = val_data[:, 0]
    y_val = prepare_targets(y_val)
    y_val = y_val.reshape((len(y_val), 1))
    y_val = tf.convert_to_tensor(y_val, dtype=np.float32)

    y_test = test_data[:, 0]
    y_test = prepare_targets(y_test)
    y_test = y_test.reshape((len(y_test), 1))
    y_test = tf.convert_to_tensor(y_test, dtype=np.float32)

    return x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test


def pad_tweets(filename, max_length, train, val, test):

    file = pd.read_csv(filename)
    data = file.values

    tokeniser = Tokenizer()
    tokeniser.fit_on_texts(data[:, 1].astype(str))
    vocab_size = len(tokeniser.word_index) + 1

    encoded_train_tweets = tokeniser.texts_to_sequences(train)
    padded_train_tweets = pad_sequences(encoded_train_tweets, maxlen=max_length, padding='post')

    encoded_val_tweets = tokeniser.texts_to_sequences(val)
    padded_val_tweets = pad_sequences(encoded_val_tweets, maxlen=max_length, padding='post')

    encoded_test_tweets = tokeniser.texts_to_sequences(test)
    padded_test_tweets = pad_sequences(encoded_test_tweets, maxlen=max_length, padding='post')

    return tokeniser, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets


def Load_GloVe(filename):

    embedding_index = dict()

    f = open(filename, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=float)
        embedding_index[word] = coefs

    print('Loaded %s word vectors' % len(embedding_index))

    return embedding_index


def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return y_enc


def get_class_weights(filename):

    file = pd.read_csv(filename)
    file['Label'] = np.where(file['Result'] == 'sarcastic', 1, 0)  #Harvested
    #file['Label'] = np.where(file['Result'] == 'SARCASM', 1, 0)    #Riloff
    sarc, non_sarc = np.bincount(file['Label'])                     #SemEval
    total = sarc + non_sarc

    weight_for_0 = (1 / non_sarc) * (total) / 2.0
    weight_for_1 = (1 / sarc) * (total) / 2.0

    initial_bias = np.log([non_sarc / sarc])
    print(initial_bias)
    initial_bias = tf.keras.initializers.Constant(initial_bias)

    return weight_for_0, weight_for_1, initial_bias


def plot_loss(loss, val_loss):

    plt.plot(loss, label='Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.show()


def plot_acc(accuracy, val_accuracy):

    plt.plot(accuracy, label='Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.show()


def roc_curve(fpr, tpr, roc_auc, title='Testing ROC Curves'):

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_roc(labels, predictions, **kwargs):

    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    roc_auc = auc(fp, tp)

    plt.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    #plt.xlim([-0.5,20])
    #plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_metrics(precision, recall):

    plt.plot(precision, label='Precision')
    plt.plot(recall, label='Recall')
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.title('Precision and Recall Performance')
    plt.show()


def plot_cm(labels, predictions, p=0.5):

    cm = confusion_matrix(labels, predictions > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show()


def plot_ypred(y_pred):
    plt.plot(y_pred, 'o')
    plt.ylabel('Predicted Labels')
    plt.xlabel('Test Tweets Numbers')
    plt.title('Predicted Labels Distribution')
    plt.show()


def plot_ypred_ytest(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('True Labels/Predicted Labels')
    plt.show()
