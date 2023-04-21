import numpy as np
import nltk
from textblob import TextBlob
from itertools import groupby


def Reshape(x):
    y = np.array(x)
    y = y.reshape((877, 1))
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

    PW = np.array(PW)
    NW = np.array(NW)
    Pos = np.array(Pos)
    Neg = np.array(Neg)
    Pos_Adj = np.array(Pos_Adj)
    Pos_Adv = np.array(Pos_Adv)
    Pos_Ver = np.array(Pos_Ver)
    Neg_Adj = np.array(Neg_Adj)
    Neg_Adv = np.array(Neg_Adv)
    Neg_Ver = np.array(Neg_Ver)

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
        pos_emo = i.count(':)') or i.count(':D') or i.count(':P') or i.count(':)') or i.count(';)')
        neg_emo = i.count(':(') or i.count(':|') or i.count(":-(")
        Excl.append(excl)
        Ques.append(ques)
        Dots.append(dots)
        Quos.append(quos)
        Caps.append(caps)
        Pos_emo.append(pos_emo)
        Neg_emo.append(neg_emo)

    Excl = np.array(Excl)
    Ques = np.array(Ques)
    Quos = np.array(Quos)
    Dots = np.array(Dots)
    Caps = np.array(Caps)
    Pos_emo = np.array(Pos_emo)
    Neg_emo = np.array(Neg_emo)

    return Excl, Ques, Quos, Dots, Caps, Pos_emo, Neg_emo


def consecutive_features(f):

    Consec_punc = []
    Consec_vowel = []

    punc = ['!', '?', '.']
    vowel = ['a', 'e', 'i', 'o', 'u']

    for i in f:
        consec_punc = [(label, sum(1 for _ in group)) for label, group in groupby(i) if label in punc]
        consec_vowel = [(label, sum(1 for _ in group)) for label, group in groupby(i) if label in vowel]

        consec_punc = [group for label, group in consec_punc if group > 1]
        consec_vowel = [group for label, group in consec_vowel if group > 1]

        Consec_punc.append(consec_punc)
        Consec_vowel.append(consec_vowel)

    Consec_punc = np.array(Consec_punc)
    Consec_vowel = np.array(Consec_vowel)

    Consec_punc = [np.sum(x) for x in Consec_punc]
    Consec_vowel = [np.sum(x) for x in Consec_vowel]

    Consec_punc = np.array(Consec_punc)
    Consec_vowel = np.array(Consec_vowel)

    return Consec_punc, Consec_vowel





