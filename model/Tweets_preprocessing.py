import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#nltk.download('stopwords')

file = pd.read_csv('Harvested.csv')
#file = file.dropna()

Stopwords = stopwords.words('english')
Stemmer = SnowballStemmer('english')

def preprocess(text, stem=False):
    text = re.sub('@\S+|@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in Stopwords:
            if stem:
                tokens.append(Stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

file['Tweets'] = file['Tweets'].apply(lambda x: preprocess(x))


seq_len = [len(i.split()) for i in file['Tweets']]
pd.Series(seq_len).hist(bins=30)
plt.title('Length of Tweets Distribution')
plt.xlabel('Length of Tweets')
plt.ylabel('Number of Tweets')
plt.show()

file.to_csv('Harvested_Cleaned.csv')
