import pandas as pd

file = pd.read_csv('#Sarcasm_Tweets.csv')
#file1 = pd.read_csv('#Sarcasm_Tweets1.csv')
#file2 = pd.read_csv('#Sarcasm_Tweets2.csv')
#file3 = pd.read_csv('#Sarcasm_Tweets3.csv')
#file4 = pd.read_csv('#Sarcasm_Tweets4.csv')
#file5 = pd.read_csv('#Sarcasm_Tweets5.csv')
#file6 = pd.read_csv('#Sarcasm_Tweets6.csv')
#file7 = pd.read_csv('#Sarcasm_Tweets7.csv')
#file8 = pd.read_csv('#Sarcasm_Tweets8.csv')
#file9 = pd.read_csv('#Sarcasm_Tweets9.csv')
#file10 = pd.read_csv('#Sarcasm_Tweets10.csv')
#file11 = pd.read_csv('#Sarcasm_Tweets11.csv')
#file12 = pd.read_csv('#Sarcasm_Tweets12.csv')
#file13 = pd.read_csv('#Sarcasm_Tweets13.csv')
#file14 = pd.read_csv('#Sarcasm_Tweets14.csv')
#file15 = pd.read_csv('#Sarcasm_Tweets15.csv')
#file16 = pd.read_csv('#Sarcasm_Tweets16.csv')
#file17 = pd.read_csv('#Sarcasm_Tweets17.csv')
#file18 = pd.read_csv('#Sarcasm_Tweets18.csv')
file19 = pd.read_csv('#Sarcasm_Tweets19.csv')


data = pd.concat([file])

data = data[['id', 'created_at', 'lang', 'text', 'result']]
data = data[data['lang'] == 'en']

print(data.head())
print(len(data))

data.to_csv('#Sarcasm_Tweets.csv')