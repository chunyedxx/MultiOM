# coding:utf-8

import tensorflow as tf
import math
from .Model import Model
import json


class Ontomap_syn(Model):

    def _transfer(self, trans_matrix, embeddings):
        return tf.matmul(trans_matrix, embeddings)

    def _calculate_1(self, h, t):
        return tf.square(h - t)

    def _calculate_2(self, loss):
        return 1 / (1 + tf.exp(loss))

    def embedding_def(self):
        config = self.get_config()

        # with open('.\pretrained_vectors\pre_ma.vec.json', "r") as f:
        #     embedding = json.load(f)
        #     ma_embeddings = tf.constant(embedding["ent_embeddings"])
        # with open('.\pretrained_vectors\pre_nci.vec.json', "r") as f:
        #     embedding = json.load(f)
        #     nci_embeddings = tf.constant(embedding["ent_embeddings"])
        self.nci_ent_embeddings = tf.get_variable(name="nci_ent_embeddings"
                                                  , shape=[config.ncienttotal, config.dimension]
                                                  , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                                                  # , initializer = nci_embeddings)
        self.ma_ent_embeddings = tf.get_variable(name="ma_ent_embeddings"
                                                 , shape=[config.maenttotal, config.dimension]
                                                 , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                                                 # , initializer = ma_embeddings)
        self.parameter_lists = {"nci_ent_embeddings": self.nci_ent_embeddings,
                                "ma_ent_embeddings": self.ma_ent_embeddings}

    def pre_instance(self):
        self.pos_n, self.pos_m = self.get_positive_syn_instance(in_batch=True)
        self.neg_n, self.neg_m = self.get_negative_syn_instance(in_batch=True)

    def projection_loss(self):
        config = self.get_config()
        pos_n_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.pos_n), [-1, config.dimension, 1])
        pos_m_e = tf.reshape(tf.nn.embedding_lookup(self.ma_ent_embeddings, self.pos_m), [-1, config.dimension, 1])
        neg_n_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.neg_n), [-1, config.dimension, 1])
        neg_m_e = tf.reshape(tf.nn.embedding_lookup(self.ma_ent_embeddings, self.neg_m), [-1, config.dimension, 1])
        _p_score = self._calculate_1(pos_n_e, pos_m_e)
        _p_score = tf.reshape(_p_score, [-1, 1, config.dimension])
        _n_score = self._calculate_1(neg_n_e, neg_m_e)
        _n_score = tf.reshape(_n_score, [-1, 1, config.dimension])
        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 1, keep_dims=True)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 1, keep_dims=True)
        p_loss = self._calculate_2(p_score)
        n_loss = self._calculate_2(n_score)
        pos_loss = tf.reduce_sum(-tf.log(p_loss))
        neg_loss = tf.reduce_sum(-tf.log(1 - n_loss))
        self.pro_loss = pos_loss + neg_loss

