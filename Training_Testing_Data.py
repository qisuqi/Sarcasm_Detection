import pandas as pd
import random
from sklearn.model_selection import train_test_split
import Data_Handler

file = pd.read_csv('Harvested_Cleaned.csv')
print(file['Result'].value_counts())
sarcasm = file[file['Result'] == 'sarcastic']
non_sarcasm = file[file['Result'] == 'non_sarcastic']

#print(sarcasm['Label'].value_counts())
#print(non_sarcasm['Label'].value_counts())

train_sarcasm, test_sarcasm = train_test_split(sarcasm, test_size=0.2)
train_not, test_not = train_test_split(non_sarcasm, test_size=0.2)
train_sarcasm1, val_sarcasm = train_test_split(train_sarcasm, test_size=0.2)
train_not1, val_not = train_test_split(train_not, test_size=0.2)

#print(train_sarcasm['Label'].value_counts())
#print(train_not['Label'].value_counts())

train = pd.concat([train_sarcasm1, train_not1])
val = pd.concat([val_sarcasm, val_not])
test = pd.concat([test_sarcasm, test_not])

train = train.sample(frac=1)
val = val.sample(frac=1)
test = test.sample(frac=1)

print(train['Result'].value_counts())
print(val['Result'].value_counts())
print(test['Result'].value_counts())

pd.DataFrame(train).to_csv(('Harvested_Training.csv'))
pd.DataFrame(val).to_csv(('Harvested_Validation.csv'))
pd.DataFrame(test).to_csv(('Harvested_Testing.csv'))

#pd.DataFrame(train).to_csv('Harvested_Training.csv')
#pd.DataFrame(val).to_csv('Harvested_Validation.csv')
#pd.DataFrame(test).to_csv('Harvested_Testing.csv')

#pd.DataFrame(train).to_csv('SemEval_Training.csv')
#pd.DataFrame(val).to_csv('SemEval_Validation.csv')
#pd.DataFrame(test).to_csv('SemEval_Testing.csv')

