import pandas as pd
import numpy as np
import string
from collections import Counter
import collections

file = pd.read_csv('Riloff_tweets_cleaned.csv')

count = lambda l1, l2: sum([1 for x in l1 if x in l2])

s = ['abbcd!!!?','aaaaaa', 'ooooooo', 'aeio']
#print(s.count('?'))
#print(count(s, set(string.punctuation)))

Excl = []
Ques = []
Dots = []
Quos = []
Caps = []
Pos_emo = []
Neg_emo = []

for i in file['tweets']:
    excl = i.count('!')
    ques = i.count('?')
    dots = i.count('.')
    quos = i.count('"') or i.count("'")
    caps = sum(1 for c in i if c.isupper())
    pos_emo = i.count(':)') or i.count(':D') or i.count(':P') or i.count(':)')
    neg_emo = i.count(':(') or i.count(':|') or i.count(":-(")
    Excl.append(excl)
    Ques.append(ques)
    Dots.append(dots)
    Quos.append(quos)
    Caps.append(caps)
    Pos_emo.append(pos_emo)
    Neg_emo.append(neg_emo)

Excl = np.array(Excl)
Excl.reshape((877, 1))

Ques = np.array(Ques)
Ques.reshape((877, 1))

Dots = np.array(Dots)
Dots.reshape((877, 1))

Quos = np.array(Quos)
Quos.reshape((877, 1))

Caps = np.array(Caps)
Caps.reshape((877, 1))


d = collections.defaultdict(int)

def count(x):
    for c in x:
        for i in c:
            d[i] += 1

#for c in file['tweets']:
 #   for i in c:
  #      d[i] += 1


for c in sorted(d, key=d.get, reverse=True):
    if c == 'a' or c == 'o' or c == 'e' or c == 'i' or c == 'u':
        if d[c] > 1:
            print(c, d[c])


