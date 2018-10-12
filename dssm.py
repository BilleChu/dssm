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


class roc_callback(Callback):
    def __init__(self, val_data, label):
        self.val_gen = val_data
        self.label = label
        self.val_reports = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.val_gen, batch_size=128)
        y_true = self.label
        val_roc = roc_auc_score(y_true , y_pred)
        print ("test auc: {}".format(val_roc))
        self.val_reports.append(val_roc)

class Feature(object):
    def __init__(self):
        self.slot_seq_len = 200
        self.test_ratio = 10
        self.total_sample = 0
        self.doc_slots = range(1, 38, 1)
        self.user_slots = range(38, 57, 1)
        self.user_inputs_data_train = {}
        self.doc_inputs_data_train = {}
        self.user_inputs_data_test = {}
        self.doc_inputs_data_test = {}
        self.train_label = []
        self.test_label = []

        for slot_id in self.doc_slots:
            self.doc_inputs_data_train[slot_id] = []
            self.doc_inputs_data_test[slot_id] = []
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
        for slot_id, features in self.doc_inputs_data_train.items():
            self.doc_inputs_data_train[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)
        for slot_id, features in self.doc_inputs_data_test.items():
            self.doc_inputs_data_test[slot_id] = pad_sequences(features, maxlen=self.slot_seq_len)

    def get_doc_user_inputs(self, sample, train):
        if (train):
            for slot_id in self.doc_slots:
                self.doc_inputs_data_train[slot_id].append(sample.get(slot_id, []))
            for slot_id in self.user_slots:
                self.user_inputs_data_train[slot_id].append(sample.get(slot_id, []))
        else:
            for slot_id in self.doc_slots:
                self.doc_inputs_data_test[slot_id].append(sample.get(slot_id, []))
            for slot_id in self.user_slots:
                self.user_inputs_data_test[slot_id].append(sample.get(slot_id, []))

    def parse(self, line):
        slot_id_features = {}
        v = line.strip().split("\t")
        if (len(v) != 2):
            return slot_id_featuresm, 0
        label = int(float(v[0]))
        for pair in v[1].split(";"):
            v_pair = pair.split(":")
            if (len(v_pair) != 2):
                continue
            f_index = int(v_pair[0])
            slot_id = int(v_pair[1])
            features = slot_id_features.get(slot_id, [])
            features.append(f_index)
            slot_id_features[slot_id] = features
        return slot_id_features, label

class Dssm(object):
    def __init__(self):
        self.slot2dim_map = {}
        self.embedding_dim = 6
        self.slot_seq_len = 200
        self.doc_slots = range(1, 38, 1)
        self.user_slots = range(38, 57, 1)

    def get_slot_dim(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                v_info = line.strip().split('\t')
                slot_id = int(v_info[0])
                sub_dim = int(v_info[1])
                self.slot2dim_map[slot_id] = sub_dim
#        print (self.slot2dim_map)

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
        doc_input_layers = {}
        doc_embedding_original_layers={}
        doc_embedding_sum_layers = {}
#        for slot_id in self.slot2dim_map.keys():
#            embedding_layers[slot_id] = Embedding(self.slot2dim_map[slot_id] + 1,
#                                                  self.embedding_dim,
#                                                  input_length = self.slot_seq_len,
#                                                  trainable=True)
        for slot_id in self.user_slots:
            if slot_id not in self.slot2dim_map:
                continue
            user_input_layers[slot_id] = Input(shape=(self.slot_seq_len,), dtype='int32')
            embedding_layers[slot_id] = Embedding(input_dim=self.slot2dim_map[slot_id] + 1,
                                                  output_dim=self.embedding_dim,
                                                  input_length=self.slot_seq_len,
                                                  mask_zero=True,
                                                  trainable=True)
            user_embedding_original_layers[slot_id] = embedding_layers[slot_id](user_input_layers[slot_id])
            user_embedding_sum_layers[slot_id] = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0], x[2]))(user_embedding_original_layers[slot_id])

        for slot_id in self.doc_slots:
            if slot_id not in self.slot2dim_map:
                continue
            doc_input_layers[slot_id] = Input(shape=(self.slot_seq_len,), dtype='int32')
            embedding_layers[slot_id] = Embedding(input_dim=self.slot2dim_map[slot_id] + 1,
                                                  output_dim=self.embedding_dim,
                                                  input_length=self.slot_seq_len,
                                                  mask_zero=True,
                                                  trainable=True)
            doc_embedding_original_layers[slot_id] = embedding_layers[slot_id](doc_input_layers[slot_id])
            doc_embedding_sum_layers[slot_id] = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0], x[2]))(doc_embedding_original_layers[slot_id])

        user_merge_layers = concatenate(list(user_embedding_sum_layers.values()))
        doc_merge_layers = concatenate(list(doc_embedding_sum_layers.values()))

        user_vec = self.tower(user_merge_layers)
        doc_vec = self.tower(doc_merge_layers)
        innerProduct = dot([user_vec, doc_vec], axes=1)
        pred = Dense(1, activation='sigmoid', use_bias=False, trainable=False)(innerProduct)
        self.model = Model(inputs=list(user_input_layers.values()) + list(doc_input_layers.values()),outputs=pred)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=0.001),
                           metrics=['acc'])
#        print (self.model.summary())
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def train(self, features):
#        print (features.user_inputs_data_train)
        print("-"*20)
#        print (features.doc_inputs_data_train)
        self.auc = roc_callback(list(features.user_inputs_data_test.values()) + list(features.doc_inputs_data_test.values()), features.test_label)
        callback_list = [self.auc]

        self.model.fit(list(features.user_inputs_data_train.values()) + list(features.doc_inputs_data_train.values()), \
                       features.train_label, batch_size=16, epochs=10, \
                       callbacks=callback_list, metrics=[auc])

if '__main__' == __name__:

    if (len(sys.argv) != 3):
        print ("python dssm.py <sample.txt> <slot_dim.txt>")
    features = Feature()
    features.build_samples(sys.argv[1])
    dssm_model = Dssm()
    dssm_model.get_slot_dim(sys.argv[2])
    dssm_model.build()
    dssm_model.train(features)









