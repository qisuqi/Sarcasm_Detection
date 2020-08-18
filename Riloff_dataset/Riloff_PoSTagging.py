import nltk
import pandas as pd
from collections import Counter
from textblob import TextBlob
import numpy as np
from operator import truediv
import math

#nltk.download('averaged_perceptron_tagger')

file = pd.read_csv('Riloff_tweets_cleaned2.csv')

dataset = file.values
x = dataset[:, -1]
x = x.astype(str)

tweets = np.array(file['tweets'])

exampleArray = ['wow freaking awesome']


def PoS_features(f):
    try:
        for item in f:

            tokenised = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenised)

            #tag = [t for w, t in tagged]
            #senti = [TextBlob(str(w)).sentiment.polarity for w, t in tagged]

            #pw = [t for w, t in tagged if TextBlob(w).sentiment.polarity > 0]
            #nw = [t for w, t in tagged if TextBlob(w).sentiment.polarity < 0]

            #pos = sum(i > 0 for i in senti)
            #neg = sum(i < 0 for i in senti)

            #Pw = pw.count('JJ') or pw.count('JJR') or pw.count('JJS') or pw.count('RB') or pw.count('RBS') or \
             #    pw.count('RBR') or pw.count('VB') or pw.count('VBG') or pw.count('VBD') or pw.count('VBZ') or \
             #    pw.count('VBP') or pw.count('VBN')

            #Nw = nw.count('JJ') or nw.count('JJR') or nw.count('JJS') or nw.count('RB') or nw.count('RBS') or \
             #    nw.count('RBR') or nw.count('VB') or nw.count('VBG') or nw.count('VBD') or nw.count('VBZ') or \
             #    nw.count('VBP') or nw.count('VBN')

            #PW.append(Pw)
            #NW.append(Nw)
            #Pos.append(pos)
            #Neg.append(neg)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()

            #for w, t in tagged:
                #word_senti = [w, t, TextBlob(w).sentiment.polarity]

                #pos = [w for w in word_senti if word_senti[2] > 0]
                #neu = [w for w in word_senti if word_senti[2] == 0]
                #neg = [w for w in word_senti if word_senti[2] < 0]

                #Pos.extend(pos)
                #Neu.extend(neu)
                #Neg.extend(neg)

    except Exception as e:
        print(e)
        pass


def Reshape(x):
    y = np.array(x)
    y.reshape((877, 1))
    return y


PW = []
NW = []
Pos = []
Neg = []

delta = 3

PoS_features(exampleArray)

PW = Reshape(PW)
NW = Reshape(NW)
Pos = Reshape(Pos)
Neg = Reshape(Neg)

lista = (delta * PW + Pos) - (delta * NW + Neg)
listb = (delta * PW + Pos) + (delta * NW + Neg)

ratio = list(map(truediv, lista, listb))
Ratio = [0 if math.isnan(x) else x for x in ratio]

Ratio = Reshape(Ratio)

print(Ratio.shape)



#Pos_Adj = [x for x in Pos if x == 'JJ' or x == 'JJR' or x == 'JJS']
#Pos_Adv = [x for x in Pos if x == 'RB' or x == 'RBS' or x == 'RBR']
#Pos_Verb = [x for x in Pos if x == 'VB' or x == 'VBD' or x == 'VBG' or x == 'VBN' or x == 'VBP' or x == 'VBZ']
#Neu_Adj = [x for x in Neu if x == 'JJ' or x == 'JJR' or x == 'JJS']
#Neu_Adv = [x for x in Neu if x == 'RB' or x == 'RBS' or x == 'RBR']
#Neu_Verb = [x for x in Neu if x == 'VB' or x == 'VBD' or x == 'VBG' or x == 'VBN' or x == 'VBP' or x == 'VBZ']
#Neg_Adj = [x for x in Neg if x == 'JJ' or x == 'JJR' or x == 'JJS']
#Neg_Adv = [x for x in Neg if x == 'RB' or x == 'RBS' or x == 'RBR']
#Neg_Verb = [x for x in Neg if x == 'VB' or x == 'VBD' or x == 'VBG' or x == 'VBN' or x == 'VBP' or x == 'VBZ']

#print('----------------------------------------------')
#print('{} Positive Sentiment Words'.format(int(np.size(Pos)/3)))
#print('{} Neutral Sentiment Words'.format(int(np.size(Neu)/3)))
#print('{} Negative Sentiment Words'.format(int(np.size(Neg)/3)))
#print('----------------------------------------------')
#print('{} Positive Adjectives'.format(np.size(Pos_Adj)),
 #     '\n {} Neutral Adjectives'.format(np.size(Neu_Adj)), '{} Negative Adjectives'.format(np.size(Neg_Adj)))
#print('----------------------------------------------')
#print('{} Positive Adverbs'.format(np.size(Pos_Adv)),
 #     '\n {} Neutral Adverbs'.format(np.size(Neu_Adv)), '{} Negative Adverbs'.format(np.size(Neg_Adv)))
#print('----------------------------------------------')
#print('{} Positive Verbs'.format(np.size(Pos_Verb)),
 #     '\n {} Neutral Verbs'.format(np.size(Neu_Verb)), '{} Negative Verbs'.format(np.size(Neg_Verb)))

#Adj = np.array(Adj)
#Adj.reshape((877, 1))

#Adv = np.array(Adv)
#Adv.reshape((877, 1))

#Ver = np.array(Ver)
#Ver.reshape((877, 1))

#Pos = np.array(Pos)
#Pos.reshape((877, 1))

#Neg = np.array(Neg)
#Neg.reshape((877, 1))

#Excl = np.array(Excl)
#Excl.reshape((877, 1))

#Ques = np.array(Ques)
#Ques.reshape((877, 1))

#Dots = np.array(Dots)
#Dots.reshape((877, 1))

#Quos = np.array(Quos)
#Quos.reshape((877, 1))

#Caps = np.array(Caps)
#Caps.reshape((877, 1))
