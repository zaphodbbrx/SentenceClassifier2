#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:54:05 2018

@author: lsm
"""
from typing import Dict
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Input, concatenate,Activation,Concatenate,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.core.common.log import get_logger
import numpy as np
import pickle

log = get_logger(__name__)

@register('cnn_model')
class CNN_classifier(KerasModel):
    
    def __init__(self, opt: Dict, **kwargs):
        self.opt = {}
        self.confident_threshold = 0.5
        self.classes = pickle.load(open(opt['classes'],'rb'))
        self.new2old = pickle.load(open(opt['new2old'],'rb'))
        self.save_path = kwargs['save_path']
        self.load_path = kwargs['load_path']
        self.in_y = kwargs['in_y']
        #self.classes = pickle.load(open(opt["classes"],'rb'))
        super().__init__(**kwargs)
        changeable_params = {
                     "metrics": ["categorical_accuracy"],
                     "optimizer": "Adam",
                     "loss": "categorical_crossentropy",
                     "dropout_power": 0.5,
                     "epochs": 250,
                     "batch_size": 64,
                     "val_every_n_epochs": 1,
                     "verbose": True,
                     "val_patience": 10,
                     "pooling_size":1}
        # Reinitializing of parameters
        for param in changeable_params.keys():
            if param in opt.keys():
                self.opt[param] = opt[param]
            else:
                self.opt[param] = changeable_params[param]
        self.model = self.cnn_model(opt)
        #self.model = self.bilstm_cnn_model(kwargs['params'])
        self.model.compile(optimizer = self.opt['optimizer'],
                           metrics = self.opt['metrics'],
                           loss = self.opt['loss'],
                           )
        if kwargs['mode']=='infer':
            self.load()

    def __call__(self, data, predict_proba=False, *args):
        """
        Infer on the given data
        Args:
            data: [list of sentences]
            predict_proba: whether to return probabilities distribution or only labels-predictions
            *args:

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.array(self.infer_on_batch(data))

        if predict_proba:
            return preds
        else:
            pl = proba2labels(preds, confident_threshold=self.confident_threshold, classes=self.classes)#[self.classes[np.argmax(preds[i,:])-1] for i in range(len(self.classes))]
            return [[y[0]] for y in pl]

    def train_on_batch(self, xa, ya):
        def add_noise(feats, labels, num_noise):
            fn = feats
            ln = labels
            for i in range(num_noise):
                noise = np.random.normal(1,0.02, feats.shape)
                noised = feats*noise
                fn = np.vstack([fn,noised])
                ln = np.vstack([ln,labels])
            return fn,ln
        
        vectors = np.array(xa)
        labels = labels2onehot(np.array(ya), classes = self.classes)
        va, la = add_noise(vectors, labels, 10)
        metrics_values = self.model.train_on_batch(va,la)
        return metrics_values
    
    
    def infer_on_batch(self, batch, labels=None):
        if labels:
            onehot_labels = labels2onehot(labels, classes=np.arange(1,20))
            metrics_values = self.model.test_on_batch(batch, onehot_labels)
            return metrics_values
        else:
            predictions = self.model.predict(batch)
            return predictions
        

    def cnn_model(self, opt):
        cnn_layers = opt['cnn_layers']
        emb_dim = opt['emb_dim']
        seq_len = opt['seq_len']
        pool_size = opt['pooling_size']
        dropout_power = opt['dropout_power']
        n_classes = opt['n_classes']

        model = Sequential()
        model.add(Conv1D(filters = cnn_layers[0]['filters'],
                         kernel_size = cnn_layers[0]['kernel_size'],
                         input_shape = (seq_len, emb_dim)))
        
        for i in range(1,len(cnn_layers)):
            model.add(Conv1D(filters = cnn_layers[i]['filters'],
                             kernel_size = cnn_layers[i]['kernel_size']))
        
        model.add(MaxPooling1D(pool_size))
        model.add(Flatten())
        model.add(Dropout(dropout_power))
        model.add(Dense(n_classes, activation='softmax'))
        
        return model

    def bilstm_cnn_model(self, params):
        """
        Build un-compiled BiLSTM-CNN
        Args:
            params: dictionary of parameters for NN
    
        Returns:
            Un-compiled model
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']),
                input_shape = (params['text_size'], params['embedding_size'])))
        for i in range(len(params['kernel_sizes_cnn'])):
            model.add(Conv1D(params['filters_cnn'][i],
                              kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPooling1D(4))
        model.add(Flatten())
        #model.add(Flatten())
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den'])))
        model.add(Activation('sigmoid'))
        return model
    
    
    def load(self):
        self.model.load_weights(self.load_path)
    def save(self):
        self.model.save_weights(self.save_path)
    
    def reset(self):
        pass
    