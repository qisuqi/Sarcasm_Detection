import time
from bert.tokenization import bert_tokenization
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from tensorflow.keras import layers
import tensorflow_hub as hub

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
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
    Data_Handler.load_and_split_dataset('SemEval_Training.csv', 'SemEval_Validation.csv', 'SemEval_Testing.csv')

BertTokenizer = bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2",
                            trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

vocab_length = len(tokenizer.vocab)
embedding_dim = 200
nb_epochs = 200
batch_size = 32
max_size = 20


def tokenise_tweets(tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))


tokenised_train_tweets = [tokenise_tweets(tweet) for tweet in x_train]
padded_train_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_train_tweets, maxlen=max_size, padding='post')

tokenised_val_tweets = [tokenise_tweets(tweet) for tweet in x_val]
padded_val_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_val_tweets, maxlen=max_size, padding='post')

tokenised_test_tweets = [tokenise_tweets(tweet) for tweet in x_test]
padded_test_tweets = keras.preprocessing.sequence.pad_sequences(tokenised_test_tweets, maxlen=max_size, padding='post')

weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('SemEval_Training.csv')
class_weight = {0: weight_for_0, 1: weight_for_1}


def create_model(hp):

    filter_size = hp.Int('filter_size', min_value=3, max_value=5, default=3)
    filter_unit = hp.Int('filter_unit', min_value=32, max_value=512, step=32)
    padding = hp.Choice('padding', ['valid', 'same'])
    init = hp.Choice('init', ['uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal'])
    dr = hp.Float('dr', 0.1, 0.5)
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    inputs = layers.Input(shape=(max_size,))
    embedding = layers.Embedding(vocab_length,
                                 embedding_dim,
                                 trainable=False)(inputs)

    x1 = layers.Conv1D(filter_unit, filter_size,
                       padding=padding,
                       kernel_initializer=init,
                       activation='relu')(embedding)
    x1 = layers.ReLU()(x1)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(filter_unit, filter_size,
                       padding=padding,
                       kernel_initializer=init,
                       activation='relu')(embedding)
    x2 = layers.ReLU()(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x3 = layers.Conv1D(filter_unit, filter_size,
                       padding=padding,
                       kernel_initializer=init,
                       activation='relu')(embedding)
    x3 = layers.ReLU()(x3)
    x3 = layers.GlobalMaxPooling1D()(x3)

    merged1 = tf.concat([x1, x2, x3], axis=1)
    drop = layers.Dropout(dr)(merged1)

    outputs = layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias)(drop)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=METRICS)

    return model


tuner = kt.BayesianOptimization(create_model, objective='val_accuracy', max_trials=nb_epochs)
tuner.search(padded_train_tweets, y_train,
             validation_data=(padded_val_tweets, y_val),
             batch_size=batch_size,
             epochs=nb_epochs,
             callbacks=[EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)])

best_params = tuner.get_best_hyperparameters(1)[0]

print('Filter Size: ', best_params.get('filter_size'),
      'Filter Unit: ', best_params.get('filter_unit'),
      'Padding: ', best_params.get('padding'),
      'Initialiser: ', best_params.get('init'),
      'Dropout Rate: ', best_params.get('dr'),
      'Learning Rate: ', best_params.get('lr'))

model = tuner.hypermodel.build(best_params)

start_time = time.time()
history = model.fit(padded_train_tweets, y_train,
                    validation_data=(padded_val_tweets, y_val),
                    epochs=nb_epochs,
                    batch_size=batch_size,
                    class_weight=class_weight,
                    callbacks=[EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)])

end_time = time.time()
finished_time = end_time-start_time
print('Finished Training with %0.2f seconds' % finished_time)

Data_Handler.plot_loss(history.history['loss'],
                       history.history['val_loss'])

Data_Handler.plot_acc(history.history['accuracy'],
                      history.history['val_accuracy'])

model.save('CNN_BERT_GS')
