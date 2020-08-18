import numpy as np
import nltk
from textblob import TextBlob


def Load_GloVe(filename):

    embedding_index = dict()

    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=float)
        embedding_index[word] = coefs

    print('Loaded %s word vectors' % len(embedding_index))

    return embedding_index


def Reshape(x):
    y = np.array(x)
    y.reshape((877, 1))
    return y


def PoS_features(f):
    Pos_Adj = []  # Positive adjectives
    Pos_Adv = []  # Positive adverbs
    Pos_Ver = []  # Positive verbs
    Neg_Adj = []  # Negative adjectives
    Neg_Adv = []  # Negative adverbs
    Neg_Ver = []  # Negative verbs
    Pos = []  # Positive words
    Neg = []  # Negative words
    PW = []  # Positive emotional words
    NW = []  # Negative emotional words

    try:
        for item in f:

            tokenised = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenised)

            tag = [w for w, t in tagged]
            senti = [TextBlob(str(w)).sentiment.polarity for w, t in tagged]

            pos = sum(i > 0 for i in senti)
            neg = sum(i < 0 for i in senti)

            pw = [t for w, t in tagged if TextBlob(w).sentiment.polarity > 0]
            nw = [t for w, t in tagged if TextBlob(w).sentiment.polarity < 0]

            pos_adj = pw.count('JJ') or pw.count('JJR') or pw.count('JJS')
            pos_adv = pw.count('RB') or pw.count('RBS') or pw.count('RBR')
            pos_ver = pw.count('VB') or pw.count('VBG') or pw.count('VBD') or pw.count('VBZ') or pw.count('VBP') \
                  or pw.count('VBN')

            neg_adj = nw.count('JJ') or nw.count('JJR') or nw.count('JJS')
            neg_adv = nw.count('RB') or nw.count('RBS') or nw.count('RBR')
            neg_ver = nw.count('VB') or nw.count('VBG') or nw.count('VBD') or nw.count('VBZ') or nw.count('VBP') \
                  or nw.count('VBN')

            Pw = pw.count('JJ') or pw.count('JJR') or pw.count('JJS') or pw.count('RB') or pw.count('RBS') or \
                 pw.count('RBR') or pw.count('VB') or pw.count('VBG') or pw.count('VBD') or pw.count('VBZ') or \
                 pw.count('VBP') or pw.count('VBN')

            Nw = nw.count('JJ') or nw.count('JJR') or nw.count('JJS') or nw.count('RB') or nw.count('RBS') or \
                 nw.count('RBR') or nw.count('VB') or nw.count('VBG') or nw.count('VBD') or nw.count('VBZ') or \
                 nw.count('VBP') or nw.count('VBN')

            PW.append(Pw)
            NW.append(Nw)
            Pos.append(pos)
            Neg.append(neg)
            Pos_Adj.append(pos_adj)
            Pos_Adv.append(pos_adv)
            Pos_Ver.append(pos_ver)
            Neg_Adj.append(neg_adj)
            Neg_Adv.append(neg_adv)
            Neg_Ver.append(neg_ver)

    except Exception as e:
        print(e)
        pass

    Pos_Adj = Reshape(Pos_Adj)
    Pos_Adv = Reshape(Pos_Adv)
    Pos_Ver = Reshape(Pos_Ver)
    Neg_Adj = Reshape(Neg_Adj)
    Neg_Adv = Reshape(Neg_Adv)
    Neg_Ver = Reshape(Neg_Ver)
    Pos = Reshape(Pos)
    Neg = Reshape(Neg)
    PW = Reshape(PW)
    NW = Reshape(NW)

    return PW, NW, Pos, Neg, Pos_Adj, Pos_Adv, Pos_Ver, Neg_Adj, Neg_Adv, Neg_Ver


def Punc_features(f):

    Excl = []  # Exclamation marks
    Ques = []  # Question marks
    Dots = []  # Periods
    Quos = []  # Quotation marks
    Caps = []  # All caps
    Pos_emo = []  # Positive emoticons
    Neg_emo = []  # Negative emoticons

    for i in f:
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

    Excl = Reshape(Excl)
    Ques = Reshape(Ques)
    Quos = Reshape(Quos)
    Dots = Reshape(Dots)
    Caps = Reshape(Caps)
    Pos_emo = Reshape(Pos_emo)
    Neg_emo = Reshape(Neg_emo)

    return Excl, Ques, Quos, Dots, Caps, Pos_emo, Neg_emo


