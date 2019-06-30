# coding:utf-8

import tensorflow as tf
import math
from .Model import Model
import json


class Ontomap(Model):

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
        bound = 6 / math.sqrt(config.dimension)
        # initializer = tf.random_uniform_initializer(minval=0, maxval=1))
        self.nci_ent_embeddings = tf.get_variable(name="nci_ent_embeddings"
                                                  , shape=[config.ncienttotal, config.dimension]
                                                  , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                                                  #   , initializer=nci_embeddings)
        self.ma_ent_embeddings = tf.get_variable(name="ma_ent_embeddings"
                                                 , shape=[config.maenttotal, config.dimension]
                                                 , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                                                 # , initializer=ma_embeddings)
        self.n2f_transfer_matrix = tf.get_variable(name="n2f_transfer_matrix", shape=[1, config.dimension * config.dimension]
                                                   , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.m2f_transfer_matrix = tf.get_variable(name="m2f_transfer_matrix", shape=[1, config.dimension * config.dimension]
                                                   , initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_lists = {"nci_ent_embeddings": self.nci_ent_embeddings,
                                "ma_ent_embeddings": self.ma_ent_embeddings,
                                "n2f_transfer_matrix": self.n2f_transfer_matrix,
                                "m2f_transfer_matrix": self.m2f_transfer_matrix}

    def pre_instance(self):
        self.pos_n, self.pos_m, self.pos_f = self.get_positive_instance(in_batch=True)
        self.neg_n, self.neg_m, self.neg_f = self.get_negative_instance(in_batch=True)

    def projection_loss(self):
        config = self.get_config()
        self.fma_ent_embeddings = self.read_fma_json(in_batch=True)
        p_n2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.pos_n))
        p_m2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.pos_m))
        n_n2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.neg_n))
        n_m2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.neg_m))
        pos_n_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.pos_n), [-1, config.dimension, 1])
        pos_m_e = tf.reshape(tf.nn.embedding_lookup(self.ma_ent_embeddings, self.pos_m), [-1, config.dimension, 1])
        pos_f_e = tf.reshape(tf.nn.embedding_lookup(self.fma_ent_embeddings, self.pos_f), [-1, config.dimension, 1])
        neg_n_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.neg_n), [-1, config.dimension, 1])
        neg_m_e = tf.reshape(tf.nn.embedding_lookup(self.ma_ent_embeddings, self.neg_m), [-1, config.dimension, 1])
        neg_f_e = tf.reshape(tf.nn.embedding_lookup(self.fma_ent_embeddings, self.neg_f), [-1, config.dimension, 1])
        pos_n2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.n2f_transfer_matrix, p_n2f_mat_ids),
                                [-1, config.dimension, config.dimension])
        pos_m2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.m2f_transfer_matrix, p_m2f_mat_ids),
                                    [-1, config.dimension, config.dimension])
        neg_n2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.n2f_transfer_matrix, n_n2f_mat_ids),
                                    [-1, config.dimension, config.dimension])
        neg_m2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.m2f_transfer_matrix, n_m2f_mat_ids),
                                    [-1, config.dimension, config.dimension])
        p_n2f_h = tf.reshape(self._transfer(pos_n2f_matrix, pos_n_e), [-1, config.dimension, 1])
        p_t = pos_f_e
        p_m2f_h = tf.reshape(self._transfer(pos_m2f_matrix, pos_m_e), [-1, config.dimension, 1])
        n_n2f_h = tf.reshape(self._transfer(neg_n2f_matrix, neg_n_e), [-1, config.dimension, 1])
        n_t = neg_f_e
        n_m2f_h = tf.reshape(self._transfer(neg_m2f_matrix, neg_m_e), [-1, config.dimension, 1])
        _p_n2f_score = self._calculate_1(p_n2f_h, p_t)
        _p_n2f_score = tf.reshape(_p_n2f_score, [-1, 1, config.dimension])
        _p_m2f_score = self._calculate_1(p_m2f_h, p_t)
        _p_m2f_score = tf.reshape(_p_m2f_score, [-1, 1, config.dimension])
        _n_n2f_score = self._calculate_1(n_n2f_h, n_t)
        _n_n2f_score = tf.reshape(_n_n2f_score, [-1, 1, config.dimension])
        _n_m2f_score = self._calculate_1(n_m2f_h, n_t)
        _n_m2f_score = tf.reshape(_n_m2f_score, [-1, 1, config.dimension])
        p_n2f_score = tf.reduce_sum(tf.reduce_mean(_p_n2f_score, 1, keep_dims=False), 1, keep_dims=True)
        p_m2f_score = tf.reduce_sum(tf.reduce_mean(_p_m2f_score, 1, keep_dims=False), 1, keep_dims=True)
        n_n2f_score = tf.reduce_sum(tf.reduce_mean(_n_n2f_score, 1, keep_dims=False), 1, keep_dims=True)
        n_m2f_score = tf.reduce_sum(tf.reduce_mean(_n_m2f_score, 1, keep_dims=False), 1, keep_dims=True)
        p_loss = self._calculate_2(p_n2f_score + p_m2f_score)
        n_loss = self._calculate_2(n_n2f_score + n_m2f_score)
        pos_loss = tf.reduce_sum(-tf.log(p_loss))
        neg_loss = tf.reduce_sum(-tf.log(1 - n_loss))
        self.pro_loss = pos_loss + neg_loss
