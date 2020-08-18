import numpy as np
import time
import math
import tensorflow as tf
import tensorflow_hub as hub
import bert
from sklearn.metrics import roc_curve, auc, classification_report
import Data_Handler, Models


def tokenise_tweets(tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))


batch_size = 32

x, y = Data_Handler.load_dataset('Riloff_tweets_cleaned2.csv')
label = Data_Handler.prepare_targets(y)

BertTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

tokenised_tweets = [tokenise_tweets(tweet) for tweet in x]

max_size = [max(len(t) for t in tokenised_tweets)]
max_size = int(max_size[0])

tweets_len = [[[tweet[t] if t < len(tweet) else 0 for t in range(max_size)], label[i], len(tweet)]
              for i, tweet in enumerate(tokenised_tweets)]

input_data = [(t[0], t[1]) for t in tweets_len]

processed_data = tf.data.Dataset.from_generator(lambda: input_data, output_types=(tf.int32, tf.int32))

batched_data = processed_data.padded_batch(batch_size, padded_shapes=((None, ), ()))

#print(next(iter(batched_data)))

num_train = round(len(input_data) * 0.8)
num_val = round(num_train * 0.2)

train_batch = math.ceil(num_train / batch_size)
val_batch = math.ceil(num_val / batch_size)

train_data = batched_data.take(train_batch)
val_data = batched_data.take(val_batch)
test_data = batched_data.skip(train_batch)

y_test = np.concatenate([y for x, y in test_data], axis=0)

vocab_length = len(tokenizer.vocab)
embedding_dim = 200
cnn_filters = 128
dnn_units = 256
dropout_rate = 0.5
nb_epochs = 5
ltsm_units = 100
max_length = 18

start_time = time.time()

model = Models.BERT(vocab_size=vocab_length, embedding_dim=embedding_dim, cnn_filters=cnn_filters,
                    dropout_rate=dropout_rate, dnn_units=dnn_units, ltsm_unit=ltsm_units)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, epochs=nb_epochs, validation_data=val_data)

train_loss, train_accuracy = model.evaluate(train_data)
test_loss, test_accuracy = model.evaluate(test_data)

print(model.summary())

end_time = time.time()

print('\nTesting Accuracy is', test_accuracy*100, 'Testing Loss is', test_loss*100, '\nTraining Accuracy is',
      train_accuracy*100, 'Training Loss is', train_loss*100, '\nTraining time is', end_time-start_time)

y_pred = model.predict(test_data)

fpr, tpr, thres = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

print(classification_report(y_test, y_pred))

Data_Handler.Training_Curve(history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                            history.history['val_accuracy'])

Data_Handler.ROC_Curve(fpr, tpr, roc_auc)
