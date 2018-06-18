#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:03:33 2018

@author: lsm
"""


from overrides import overrides

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from TextNormalizer import TextNormalizer

log = get_logger(__name__)

@register('text_normalizer')
class TextCorrector(Component, Serializable):
    def __init__(self, **kwargs):
        self.tn = TextNormalizer().fit()
        self.mode=kwargs['mode']
        #super().__init__(**kwargs)

    def save(self, *args, **kwargs):
        raise NotImplementedError
        
    def load(self, *args, **kwargs):
        pass

    @overrides
    def __call__(self, texts, *args, **kwargs):
        
        return self.tn.transform(texts)
