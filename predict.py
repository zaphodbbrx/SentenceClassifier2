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


class ipavlov_interact():
    def __init__(self, config_path = 'tns_config.json'):
        config = read_json(config_path)
        self.model = build_model_from_config(config)
    
    def predict(self, input_text):
        in_s = []
        in_s.append('{}::'.format(input_text))
        return {'decision':self.model(in_s)[0][0]}


if __name__ == '__main__':
    while True:
        ic = ipavlov_interact()
        mes = input()
        print(ic.predict(mes)['decision'])