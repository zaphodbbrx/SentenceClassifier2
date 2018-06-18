#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:32:17 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import *
from text_normalizer import *
from embedder import *
from augmenter import *
from CNN_model import *
from label_transformer import *
import pandas as pd
import numpy as np


class ipavlov_interact():
    def __init__(self, config_path = 'tns_config.json'):
        config = read_json(config_path)
        self.model = build_model_from_config(config)
    
    def predict(self, input_text):
        rp = self.model.pipe[0][-1]([input_text])
        for i in range(1,len(self.model.pipe)-1):
            rp = self.model.pipe[i][-1](rp)
        res = self.model.pipe[-1][-1](rp, predict_proba = True)
        dec = proba2labels(res, confident_threshold=self.model.pipe[-1][-1].confident_threshold, classes=self.model.pipe[-1][-1].classes)[0][0]
        return {
            'decision': dec,
            'confidence': np.max(res)
               }


if __name__ == '__main__':
    while True:
        ic = ipavlov_interact()
        mes = input()
        print(ic.predict(mes))