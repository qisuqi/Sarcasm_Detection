import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import Data_Handler


x, y, _ = Data_Handler.load_dataset('SemEval_cleaned.csv')

bigram_train_vec = CountVectorizer(ngram_range=(1, 2))
bigram_train_words = bigram_train_vec.fit_transform(x)

#transformer = TfidfTransformer(norm=None, smooth_idf=False, sublinear_tf=False, use_idf=True)
#tfidf_train_words = transformer.fit_transform(bigram_train_words)-bigram_train_words

#kf = KFold(n_splits=10)

model = SVC(gamma='scale')

Error = []
Accuracy = []
Precision = []
Recall = []

#for train_index, test_index in kf.split(bigram_train_words):

#x_train, x_test = bigram_train_words[train_index], bigram_train_words[test_index]
#y_train, y_test = y[train_index], y[test_index]

    #Error.append(error)
    #Accuracy.append(accuracy)
    #Precision.append(precision)
    #Recall.append(recall)

start_time = time.time()

X_train, X_test, Y_train, Y_test = train_test_split(bigram_train_words, y, test_size=0.2, random_state=0)

classifier = model.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

n = X_test.shape[0]
error = np.sum(Y_test != y_pred)
error = error * 100
error = error/float(n)
accuracy = 100 - error

precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)

clf = model.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

end_time = time.time()

print('Final Error is', np.mean(error), 'Final Accuracy is ', np.mean(accuracy),
      'Training time is', end_time-start_time)


#Data_Handler.error_accuracy(Error, Accuracy)

Data_Handler.plot_metrics(Precision, Recall)

Data_Handler.plot_cm(Y_test, Y_pred)

Data_Handler.plot_roc(Y_test, Y_pred)

Data_Handler.plot_ypred(Y_pred)

Data_Handler.plot_ypred_ytest(Y_test, Y_pred)

Y_pred[Y_pred <= 0.5] = 0.
Y_pred[Y_pred > 0.5] = 1.

print(classification_report(Y_test, Y_pred))





