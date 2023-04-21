import time
from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import kerastuner as kt
import matplotlib.pyplot as plt
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


weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('SemEval_Training.csv')

class_weight = {0: weight_for_0, 1: weight_for_1}

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
nb_epochs = 100
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

early_stopping = EarlyStopping(
    monitor='val_auc',
    verbose=0,
    patience=10,
    mode='max',
    restore_best_weights=True)


def create_model(hp):

    embd_init = hp.Choice('emb_init', ['uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal'])
    blstm_init = hp.Choice('blstm_init', ['uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal'])
    blstm_init1 = hp.Choice('blstm1_init', ['uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal'])
    blstm_reg = hp.Choice('blstm_reg', [])

    inputs = layers.Input(shape=(max_size,))
    embedding = layers.Embedding(vocab_length,
                                 embedding_dim,
                                 trainable=False,
                                 embeddings_initializer=embd_init)(inputs)
    drop = layers.Dropout(hp.Float('emb_dr', 0.1, 0.5))(embedding)
    blstm = layers.Bidirectional(layers.LSTM(100,
                                             return_sequences=True,
                                             activation='tanh',
                                             recurrent_activation='sigmoid',
                                             dropout=hp.Float('blstm_dr', 0.1, 0.5),
                                             recurrent_dropout=hp.Float('rec_dr', 0.1, 0.5),
                                             kernel_initializer=blstm_init,
                                             recurrent_initializer=blstm_init1,
                                             kernel_regularizer='l2',
                                             recurrent_regularizer='l2'))(drop)
    att = Models.SelfAttention(size=64, num_hops=20, use_penalise=False)(blstm)

    inputs2 = layers.Input(shape=(max_size, 1))
    merged = tf.concat([att, inputs2], axis=-1)

    filter_units = hp.Int('FilterNum', min_value=32, max_value=512, step=32)
    filter_size = hp.Int('FilterSize', min_value=3, max_value=5, default=3)
    conv_init = hp.Choice('conv_init', ['uniform', 'normal', 'he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal'])

    x1 = layers.Conv1D(filter_units, filter_size,
                       padding='valid',
                       kernel_initializer=conv_init,
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.Conv1D(filter_units, filter_size,
                       padding='valid',
                       kernel_initializer=conv_init,
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x3 = layers.Conv1D(filter_units, filter_size,
                       padding='valid',
                       kernel_initializer=conv_init,
                       kernel_regularizer='l2',
                       activation='relu')(merged)
    x3 = layers.GlobalMaxPooling1D()(x3)

    merged1 = tf.concat([x1, x2, x3], axis=1)
    dense = layers.Dense(hp.Int('dense', min_value=32, max_value=512, step=32), activation='relu')(merged1)
    drop = layers.Dropout(hp.Float('conv_dr', 0.1, 0.5))(dense)

    dense1 = layers.Dense(1,
                          activation='sigmoid',
                          bias_initializer=initial_bias,
                          kernel_regularizer='l2')(drop)

    model = keras.Model(inputs=[inputs, inputs2], outputs=dense1)

    #print(model.summary())

    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=METRICS)

    return model


tuner = kt.BayesianOptimization(create_model, objective='val_accuracy', max_trials=nb_epochs)
tuner.search([padded_train_tweets, aux_train], y_train,
             validation_data=([padded_val_tweets, aux_val], y_val),
             batch_size=batch_size,
             epochs=nb_epochs,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc', restore_best_weights=True)])
best_params = tuner.get_best_hyperparameters(1)[0]

print('Embedding Initialiser: ', best_params.get('emb_init'),
      'Embedding Drop', best_params.get('emb_dr'),
      'BLSTM Initialiser:', best_params.get('blstm_init'),
      'BLSTM1 Initialiser:', best_params.get('blstm1_init'),
      'BLSTM Dropout: ', best_params.get('blstm_dr'),
      'BLSTM1 Dropout: ', best_params.get('rec_dr'),
      'Filter Number:', best_params.get('FilterNum'),
      'Filter Size:', best_params.get('FilterSize'),
      'Conv Initialiser: ', best_params.get('conv_init'),
      'Dense Units:', best_params.get('dense'),
      'Learning Rate:', best_params.get('lr'),
      'Conv Drop: ', best_params.get('conv_dr')
      )

model = tuner.hypermodel.build(best_params)
history = model.fit([x_train, aux_train], y_train,
                    validation_data=([x_val, aux_val], y_val),
                    epochs=nb_epochs,
                    batch_size=batch_size,
                    class_weight=class_weight,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc',
                                                                restore_best_weights=True)])

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

y_pred = model.predict([x_test, aux_test], batch_size=batch_size)

print(metrics.classification_report(y_test, y_pred.round()))

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)

