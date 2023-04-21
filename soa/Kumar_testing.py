import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
import tensorflow.keras as keras
import Data_Handler

max_length = 20
batch_size = 128

x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    Data_Handler.load_and_split_dataset('Harvested_Training.csv', 'Harvested_Validation.csv', 'Harvested_Testing.csv')

y_test = keras.utils.to_categorical(y_test)

tokeniser, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    Data_Handler.pad_tweets('Harvested_Cleaned.csv', max_length, x_train, x_val, x_test)

model = keras.models.load_model('Kumar_Harvested')

test_result = model.evaluate([padded_test_tweets, aux_test], y_test)

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

y_pred = model.predict([padded_test_tweets, aux_test])

y_test = np.argmax(y_test, axis=-1)
y_pred = np.argmax(y_pred, axis=-1)

print(classification_report(y_test, y_pred))

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)
