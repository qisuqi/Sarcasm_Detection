# Import relevant pacakges
import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import classification_report
# Helper Functions
import Data_Handler, Models

# Define the parameters
max_length = 20
batch_size = 64

# Split the datasets to tweets, auxiliary features, and their labels
# For the Harvested dataset
x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    Data_Handler.load_and_split_dataset('Harvested_Training.csv', 'Harvested_Validation.csv', 'Harvested_Testing.csv')
# For the SemEval dataset
#x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    #Data_Handler.load_and_split_dataset('SemEval_Training.csv', 'SemEval_Validation.csv', 'SemEval _Testing.csv')
# For the Riloff dataset
#x_train, x_val, x_test, aux_train, aux_val, aux_test, y_train, y_val, y_test = \
    #Data_Handler.load_and_split_dataset('Riloff_Training.csv', 'Riloff_Validation.csv', 'Riloff _Testing.csv')


# Convert the label to be adaptable for using softmax classifier
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

# Perform zero padding with the threshold value
# For the Harvested dataset
t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    Data_Handler.pad_tweets('Harvested_Cleaned.csv', max_length, x_train, x_val, x_test)
# For the SemEval dataset
#t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    #Data_Handler.pad_tweets('SemEval_Cleaned.csv', max_length, x_train, x_val, x_test)
# For the Riloff dataset
#t, vocab_size, padded_train_tweets, padded_val_tweets, padded_test_tweets = \
    #Data_Handler.pad_tweets('Riloff_Cleaned.csv', max_length, x_train, x_val, x_test)

# Load the saved model and specify the customer layer
model = keras.models.load_model('af_sAtt_BLSTM_CNN_Harvested.h5',
                                custom_objects={'SelfAttention': Models.SelfAttention})

# Evaluate the model on testing data
test_result = model.evaluate([padded_test_tweets, aux_test], y_test)


# Make the predication using testing data
y_pred = model.predict([padded_test_tweets, aux_test])

# Use the argmax function to cover the true and predicted labels to a single value
y_test = np.argmax(y_test, axis=-1)
y_pred = np.argmax(y_pred, axis=-1)

# Show the classification report
print(classification_report(y_test, y_pred))

# Visualise the confusion matrix
Data_Handler.plot_cm(y_test, y_pred)

# Visualise the roc curve
Data_Handler.plot_roc(y_test, y_pred)