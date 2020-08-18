import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


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


def split_dataset(x, y):

    val_split = 0.2

    indices = np.arange(x.shape[0])
    x = x[indices]
    label1 = y[indices]
    label = prepare_targets(label1)
    num_val = int(val_split * x.shape[0])

    x_train = x[:-num_val]
    y_train = label[:-num_val]
    x_val = x_train[-num_val:]
    y_val = y_train[-num_val:]
    x_test = x[-num_val:]
    y_test = label[-num_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def split_features(x):
    val_split = 0.2

    indices = np.arange(x.shape[0])
    x = x[indices]

    num_val = int(val_split * x.shape[0])

    fea_train = x[:-num_val]
    fea_val = fea_train[-num_val:]
    fea_test = x[-num_val:]

    return fea_train, fea_val, fea_test



def Training_Curve(loss, accuracy, val_loss, val_accuracy):
    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.show()


def ROC_Curve(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

def Error_Accuracy(loss, accuracy):
    plt.plot(loss, label='Error')
    plt.plot(accuracy, label='Accuracy')
    plt.title('10-Fold N-gram Baseline Model')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy/Error')
    plt.legend()
    plt.show()
