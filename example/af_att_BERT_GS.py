import numpy as np
import time
import warnings
from bert.tokenization import bert_tokenization
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import kerastuner as kt
from sklearn.metrics import classification_report
import Data_Handler, Models, Features

warnings.filterwarnings('ignore')

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


weight_for_0, weight_for_1, initial_bias = Data_Handler.get_class_weights('SemEval_cleaned.csv')

class_weight = {0: weight_for_0, 1: weight_for_1}

x, y, aux = Data_Handler.load_dataset('SemEval_cleaned.csv')

BertTokenizer = bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


def tokenise_tweets(tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))


tokenised_tweets = [tokenise_tweets(tweet) for tweet in x]

#max_size = [max(len(t) for t in tokenised_tweets)]
#max_size = int(max_size[0])
max_size = 15

tweets_len = [[[tweet[t] if t < len(tweet) else 0 for t in range(max_size)], y[i], aux[i], len(tweet)]
              for i, tweet in enumerate(tokenised_tweets)]


input_data = [(t[0], t[1], t[2]) for t in tweets_len]
input_data = np.array(input_data)

x_train, y_train, x_val, y_val, x_test, y_test, aux_train, aux_val, aux_test = Data_Handler.split_features(input_data)
print(x_train.shape, y_train.shape, aux_train.shape)

vocab_length = len(tokenizer.vocab)
embedding_dim = 200
nb_epochs = 100
batch_size = 32

def create_model(hp):

    inputs = layers.Input(shape=(max_size,))
    embedding = layers.Embedding(vocab_length,
                                 embedding_dim,
                                 embeddings_initializer=hp.Choice('emb_init',
                                                                  ['uniform', 'normal', 'he_normal', 'he_uniform']))(inputs)

    blstm_units = hp.Int('BLSTM', min_value=100, max_value=500, step=100)

    bltsm1 = layers.LSTM(units=blstm_units,
                         return_sequences=True,
                         activation=hp.Choice('lstm_act', ['relu', 'tanh', 'sigmoid']),
                         kernel_initializer=hp.Choice('lstm_init', ['uniform', 'normal', 'he_normal', 'he_uniform']))
    bltsm2 = layers.LSTM(units=blstm_units,
                         return_sequences=True,
                         go_backwards=True,
                         activation=hp.Choice('lstm1_act', ['relu', 'tanh', 'sigmoid']),
                         kernel_initializer=hp.Choice('lstm1_init', ['uniform', 'normal', 'he_normal', 'he_uniform']))
    bltsm = layers.Bidirectional(bltsm1, backward_layer=bltsm2)(embedding)

    att = Models.SelfAttention(size=128, num_hops=20, use_penalise=False)(bltsm)

    inputs2 = layers.Input(shape=(20, 1))

    merged = tf.concat([att, inputs2], axis=-1)

    filter_units = hp.Int('FilterNum', min_value=32, max_value=512, step=32)
    filter_size = hp.Int('FilterSize', min_value=3, max_value=5, default=3)

    conv1 = layers.Conv1D(filters=filter_units,
                          kernel_size=filter_size,
                          padding=hp.Choice('padding', ['valid', 'same']),
                          kernel_initializer=hp.Choice('conv_init', ['uniform', 'normal', 'he_normal', 'he_uniform']),
                          activation='relu')(merged)
    conv2 = layers.Conv1D(filters=filter_units*2,
                          kernel_size=filter_size,
                          padding=hp.Choice('padding1', ['valid', 'same']),
                          kernel_initializer=hp.Choice('conv1_init', ['uniform', 'normal', 'he_normal', 'he_uniform']),
                          activation='relu')(conv1)

    relu = layers.ReLU()(conv2)

    dr = hp.Float('dr', 0.1, 0.2)

    dropout = layers.Dropout(dr)(relu)

    maxpool = layers.GlobalMaxPooling1D()(dropout)

    #flat = layers.Flatten()(merged)

    dense_units = hp.Int('Dense', min_value=32, max_value=128, step=32)

    dense = layers.Dense(units=dense_units, activation='relu')(maxpool)

    outputs = layers.Dense(1, activation='sigmoid')(dense)

    model = keras.Model(inputs=[inputs, inputs2], outputs=[outputs])

    model.compile(keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss='binary_crossentropy',
                  metrics=METRICS)
    return model


tuner = kt.BayesianOptimization(create_model, objective='val_accuracy', max_trials=50)

tuner.search([x_train, aux_train], y_train, validation_data=([x_val, aux_val], y_val), epochs=50,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc', restore_best_weights=True)])

#best_model = tuner.get_best_models(1)[0]
best_params = tuner.get_best_hyperparameters(1)[0]

print('Embedding Initialiser: ', best_params.get('emb_init'),
      'BLSTM:', best_params.get('BLSTM'),
      'LSTM Activation: ', best_params.get('lstm_act'),
      'LSTM1 Activation: ', best_params.get('lstm1_act'),
      'LSTM Initialiser: ', best_params.get('lstm_init'),
      'LSTM1 Initaliser: ', best_params.get('lstm1_init'),
      'Filter Number:', best_params.get('FilterNum'),
      'Filter Size:', best_params.get('FilterSize'),
      'Padding: ', best_params.get('padding'),
      'Padding1: ', best_params.get('padding1'),
      'Conv Initialiser: ', best_params.get('conv_init'),
      'Conv1 Initialiser: ', best_params.get('conv1_init'),
      'Dense Units:', best_params.get('Dense'),
      'Learning Rate:', best_params.get('lr'),
      'Dropout Rate: ', best_params.get('dr')
      )

model = tuner.hypermodel.build(best_params)
history = model.fit([x_train, aux_train], y_train, validation_data=([x_val, aux_val], y_val),
                    epochs=50, class_weight=class_weight,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc',
                                                                restore_best_weights=True)])
test_result = model.evaluate([x_test, aux_test], y_test)
print(test_result)

y_pred = model.predict([x_test, aux_test])

Data_Handler.training_curve(history.history['loss'], history.history['accuracy'])#, history.history['val_loss'],
                            #history.history['val_accuracy'])

Data_Handler.plot_metrics(history.history['precision'], history.history['recall'])

Data_Handler.plot_cm(y_test, y_pred)

Data_Handler.plot_roc(y_test, y_pred)

Data_Handler.plot_ypred(y_pred)

Data_Handler.plot_ypred_ytest(y_test, y_pred)

y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.

print(classification_report(y_test, y_pred))
