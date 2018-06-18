#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:40:06 2018

@author: lsm
"""
import sys
from overrides import overrides

import numpy as np
import pickle

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable

log = get_logger(__name__)

@register('embedder')
class SentenceEmbedder(Component, Serializable):
    def __init__(self, save_path=None, load_path = None, dim=50, **kwargs):
        super().__init__(save_path=save_path, load_path=load_path)
        self.dim = dim
        self.model = self.load()
        self.mode=kwargs['mode']

    def save(self, *args, **kwargs):
        raise NotImplementedError
        
    def load(self, *args, **kwargs):
        model = pickle.load(open('ft_compressed.pkl','rb'))
        return model
    
    def __make_padded_sequences(self,docs, max_length,w2v):
        tokens = [doc.split(' ') for doc in docs]
        vecs = [[w2v[t] if t in w2v else np.zeros(25) for t in ts] for ts in tokens]
        seqs = np.array([np.pad(np.vstack(v),mode = 'constant', pad_width = ((0,max_length-len(v)),(0,0))) if len(v)<max_length else np.vstack(v)[:max_length,:] for v in vecs])
        return seqs
    
    @overrides
    def __call__(self, batch, mean=False, *args, **kwargs):
        seqs = self.__make_padded_sequences(batch, max_length = 100, w2v = self.model)
        return seqs
    
    
    
        