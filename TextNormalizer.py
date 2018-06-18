#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import pymystem3
from nltk.tokenize import TreebankWordTokenizer
import re
class TextNormalizer(BaseEstimator,TransformerMixin):
    def __init__(self, vocab_filename = None):
        self.__tokenizer = TreebankWordTokenizer()
        self.__mystem = pymystem3.Mystem()
        
    
    def __clean_comment(self, text):
        text = str(text)
        if len(text)>0:
            text = re.sub('\W|\d',' ',text).lower()
            tokens = self.__tokenizer.tokenize(text)
            tokens = [self.__mystem.lemmatize(t)[0] for t in tokens]
            return ' '.join(tokens)
    
    def transform(self, X, y=None, **fit_params):
        res = []
        for line in X:
            res.append(self.__clean_comment(line))
        return res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X=None, y=None, **fit_params):
        return self
    
if __name__ == '__main__':
    tn = TextNormalizer()
    print(tn.fit_transform(['првет дург']))