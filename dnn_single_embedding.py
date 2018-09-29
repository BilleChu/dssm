#! /usr/bin/python
#! coding: utf8

import os
import sys
import keras
import random

import numpy as np
import tensorflow as tf


from keras import optimizers
from keras.models import Model
from keras.activations import sigmoid
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Flatten, Input, Lambda, concatenate, dot, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_auc_score
from keras.utils.vis_utils import plot_model
import keras.backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
K.set_session(session)

class Feature(object):
    def __init__(self, length):
        self.slot_seq_len = 1
        self.test_ratio = 10
        self.total_sample = 0
        self.user_slots = range(1, length, 1)
        self.user_inputs_data_train = {}
        self.user_inputs_data_test = {}
        self.train_label = []
        self.test_label = []

        for slot_id in self.user_slots:
            self.user_inputs_data_train[slot_id] = []
            self.user_inputs_data_test[slot_id] = []

    def build_samples(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.total_sample += 1
                if (self.total_sample % self.test_ratio):
                    train_temp, label= self.parse(line)
                    self.get_doc_user_inputs(train_temp, True)
                    self.train_label.append(label)
                else:
                    test_temp, label = self.parse(line)
                    self.get_doc_user_inputs(test_temp, False)
                    self.test_label.append(label)

        for slot_id, features in self.user_inputs_data_train.items():
            self.user_inputs_data_train[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)
        for slot_id, features in self.user_inputs_data_test.items():
            self.user_inputs_data_test[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)

    def build_samples(self, filename, TRAIN):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.total_sample += 1
                if (TRAIN):
                    train_temp, label= self.parse(line)
                    self.get_doc_user_inputs(train_temp, True)
                    self.train_label.append(label)
                else:
                    test_temp, label = self.parse(line)
                    self.get_doc_user_inputs(test_temp, False)
                    self.test_label.append(label)
        if (TRAIN):
            for slot_id, features in self.user_inputs_data_train.items():
                self.user_inputs_data_train[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)
        else:
            for slot_id, features in self.user_inputs_data_test.items():
                self.user_inputs_data_test[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)


    def get_doc_user_inputs(self, sample, train):
        if (train):
            for slot_id in self.user_slots:
                self.user_inputs_data_train[slot_id].append(sample.get(slot_id, []))
        else:
            for slot_id in self.user_slots:
                self.user_inputs_data_test[slot_id].append(sample.get(slot_id, []))

    def parse(self, line):
        slot_id_features = {}
        v = line.strip().split(" ")
        label = int(v[0])
        for pair in v[1:]:
            v_pair = pair.split(":")
            if (len(v_pair) != 3):
                continue
            f_index = int(v_pair[1])
            slot_id = int(v_pair[0])
            features = slot_id_features.get(slot_id, [])
            features.append(f_index)
            slot_id_features[slot_id] = features
        return slot_id_features, label

class Dnn(object):
    def __init__(self):
        self.user_slots = range(0, 39, 1)
        self.embedding_dim = 8
        self.slot_seq_len = 1
        self.embedding_length = 1000000

    def tower(self, merge_layers):
        x = Dense(512, activation='relu', kernel_initializer='random_uniform')(merge_layers)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_initializer='random_uniform')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu', kernel_initializer='random_uniform')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu', kernel_initializer='random_uniform')(x)
        x = BatchNormalization()(x)
        return x

    def build(self):
        print ('begin to build model')
        embedding_layers = {}
        user_input_layers = {}
        user_embedding_original_layers={}
        user_embedding_sum_layers={}
        embedding_layers = Embedding(input_dim=self.embedding_length + 1,
                                     output_dim=self.embedding_dim,
                                     input_length=self.slot_seq_len,
                                     mask_zero=True,
                                     trainable=True)

        for slot_id in self.user_slots:
            user_input_layers[slot_id] = Input(shape=(self.slot_seq_len,), dtype='int32')
            user_embedding_original_layers[slot_id] = embedding_layers(user_input_layers[slot_id])
            user_embedding_sum_layers[slot_id] = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0], x[2]))(user_embedding_original_layers[slot_id])

        user_merge_layers = concatenate(list(user_embedding_sum_layers.values()))
        user_vec = self.tower(user_merge_layers)
        pred = Dense(1, activation='sigmoid', use_bias=False, trainable=False)(user_vec)
        self.model = Model(inputs=list(user_input_layers.values()),outputs=pred)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=0.001),
                           metrics=['acc'])
#        print (self.model.summary())
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def train(self, features):
        print("-"*20)
        self.model.fit(list(features.user_inputs_data_train.values()), \
                       features.train_label, batch_size=16, epochs=10, \
                       validation_data=(list(features.user_inputs_data_test.values()), \
                       features.test_label))

if '__main__' == __name__:

    if (len(sys.argv) != 3):
        print ("python dssm.py <train_sample.txt> <test_sample.txt>")
    dnn_model = Dnn()
    features = Feature(len(dnn_model.user_slots) + 1)
    features.build_samples(sys.argv[1], True)
    features.build_samples(sys.argv[2], False)
    dnn_model.build()
    dnn_model.train(features)









