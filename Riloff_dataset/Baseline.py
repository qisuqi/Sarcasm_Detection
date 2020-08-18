import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report
import Data_Handler

x, y = Data_Handler.load_dataset('Riloff_tweets_cleaned2.csv')
label = Data_Handler.prepare_targets(y)

bigram_train_vec = CountVectorizer(ngram_range=(1, 2))
bigram_train_words = bigram_train_vec.fit_transform(x)

#transformer = TfidfTransformer(norm=None, smooth_idf=False, sublinear_tf=False, use_idf=True)
#tfidf_train_words = transformer.fit_transform(bigram_train_words)-bigram_train_words

kf = KFold(n_splits=10)

model = SVC(gamma='scale')

Error = []
Accuracy = []
TPR = []
FPR = []

start_time = time.time()
for train_index, test_index in kf.split(bigram_train_words):

    x_train, x_test = bigram_train_words[train_index], bigram_train_words[test_index]
    y_train, y_test = label[train_index], label[test_index]

    classifier = model.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    n = x_test.shape[0]
    error = np.sum(y_test != y_pred)
    error = error * 100
    error = error/float(n)
    accuracy = 100 - error

    Error.append(error)
    Accuracy.append(accuracy)

end_time = time.time()

print('Final Error is', np.mean(Error), 'Final Accuracy is ', np.mean(Accuracy),
      'Training time is', end_time-start_time)

X_train, X_test, Y_train, Y_test = train_test_split(bigram_train_words, label, test_size=0.2, random_state=0)

clf = model.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

fpr, tpr, thres = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)

Y_pred[Y_pred <= 0.5] = 0.
Y_pred[Y_pred > 0.5] = 1.

print(classification_report(Y_test, Y_pred))

Data_Handler.Error_Accuracy(Error, Accuracy)

Data_Handler.ROC_Curve(fpr, tpr, roc_auc)





