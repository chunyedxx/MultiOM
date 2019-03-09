# coding:utf-8

import tensorflow as tf
import math
from .Model import Model


class Ontomap(Model):

    def _transfer(self, trans_matrix, embeddings):
        return tf.matmul(trans_matrix, embeddings)

    def _calculate(self, h, t):
        return tf.square(h - t)

    def embedding_def(self):
        config = self.get_config()
        bound = 6 / math.sqrt(config.dimension)
        # initializer = tf.random_uniform_initializer(minval=-bound, maxval=bound, seed=123))
        self.nci_ent_embeddings = tf.random_uniform(shape=[config.ncienttotal, config.dimension], name= 'nci_ent_embeddings')
        self.ma_ent_embeddings = tf.random_uniform(shape=[config.maenttotal, config.dimension], name= 'ma_ent_embeddings')
        self.fma_ent_embeddings = tf.random_uniform(shape=[config.fmaenttotal, config.dimension], name= 'fma_ent_embeddings')
        self.n2f_transfer_matrix = tf.get_variable(name="n2f_transfer_matrix", shape=[1, config.dimension * config.dimension],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.m2f_transfer_matrix = tf.get_variable(name="m2f_transfer_matrix", shape=[1, config.dimension * config.dimension],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_lists = {"n2f_transfer_matrix": self.n2f_transfer_matrix,
                                "m2f_transfer_matrix": self.m2f_transfer_matrix,
                                "nci_ent_embeddings":self.nci_ent_embeddings,
                                "ma_ent_embeddings": self.ma_ent_embeddings,
                                "fms_ent_embeddings": self.fma_ent_embeddings}

        bound = 6 / math.sqrt(config.dimension)

    def pre_instance(self):

        self.pos_n, self.pos_m, self.pos_f = self.get_positive_instance(in_batch=True)

    def projection_loss(self):
        config = self.get_config()

        p_n2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.pos_n))
        p_m2f_mat_ids = tf.zeros(dtype=tf.int64, shape=tf.shape(self.pos_m))
        pos_n_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.pos_n), [-1, config.dimension, 1])
        pos_m_e = tf.reshape(tf.nn.embedding_lookup(self.nci_ent_embeddings, self.pos_m), [-1, config.dimension, 1])
        pos_f_e = tf.reshape(tf.nn.embedding_lookup(self.fma_ent_embeddings, self.pos_f), [-1, config.dimension, 1])
        pos_n2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.n2f_transfer_matrix, p_n2f_mat_ids),
                                [-1, config.dimension, config.dimension])
        pos_m2f_matrix = tf.reshape(tf.nn.embedding_lookup(self.m2f_transfer_matrix, p_m2f_mat_ids),
                                    [-1, config.dimension, config.dimension])
        p_n2f_h = tf.reshape(self._transfer(pos_n2f_matrix, pos_n_e), [-1, config.dimension,1])
        p__t = pos_f_e
        p_m2f_h = tf.reshape(self._transfer(pos_m2f_matrix, pos_m_e), [-1, config.dimension, 1])
        _p_n2f_score = self._calculate(p_n2f_h, p__t)
        _p_n2f_score = tf.reshape(_p_n2f_score, [-1, 1, config.dimension])
        _p_m2f_score = self._calculate(p_m2f_h, p__t)
        _p_m2f_score = tf.reshape(_p_m2f_score, [-1, 1, config.dimension])
        p_n2f_score = tf.reduce_sum(tf.reduce_mean(_p_n2f_score, 1, keep_dims=False), 1, keep_dims=True)
        n2f_loss = tf.reduce_sum(p_n2f_score)
        p_m2f_score = tf.reduce_sum(tf.reduce_mean(_p_m2f_score, 1, keep_dims=False), 1, keep_dims=True)
        m2f_loss = tf.reduce_sum(p_m2f_score)
        self.pro_loss = n2f_loss + m2f_loss


