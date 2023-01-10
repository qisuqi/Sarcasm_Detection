import pandas as pd
import numpy as np
import re
import Features

file = pd.read_csv('Harvested_Cleaned.csv')


def remove_url(text):
    text = re.sub('@\S+|@\S+|https?:\S+|http?:\S+|#\S+|#\S', ' ', str(text)).strip()
    return text


#file['Tweets'] = file['Tweets'].apply(lambda x: remove_url(x))

#file = file.drop(columns='Unnamed: 0')
#file.to_csv('Riloff_Cleaned2.csv')


delta = np.ones(shape=(2000, ))
delta = delta*3

PW, NW, Pos, Neg, Pos_Adj, Pos_Adv, Pos_Ver, Neg_Adj, Neg_Adv, Neg_Ver = Features.PoS_features(file['Tweets'])
Excl, Ques, Quos, Dots, Caps, Pos_emo, Neg_emo = Features.Punc_features(file['Tweets'])
Consec_punc, Consec_vowel = Features.consecutive_features(file['Tweets'])

lista = (delta * PW + Pos) - (delta * NW + Neg)
listb = (delta * PW + Pos) + (delta * NW + Neg)

ratio = lista/listb
Ratio = np.nan_to_num(ratio)

aux = np.array([PW, NW, Pos, Neg, Pos_Adj, Pos_Adv, Pos_Ver, Neg_Adj, Neg_Adv, Neg_Ver, Excl, Ques, Dots, Quos, Caps,
                Ratio, Pos_emo, Neg_emo, Consec_punc, Consec_vowel]).T

#Aux = np.expand_dims(aux, axis=-1)

pd.DataFrame(aux).to_csv('a.csv')
#print(aux)

