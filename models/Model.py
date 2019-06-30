# coding:utf-8
import numpy as np
import tensorflow as tf
import json


class Model(object):

    def get_config(self):
        return self.config

    def read_fma_json(self, in_batch=True):
        if in_batch:
            with open('.\pretrained_vectors\\fma_conve.json', "r") as f:
                embedding = json.load(f)
                self.fma_embeddings = tf.constant(embedding["ent_embeddings"])
        return self.fma_embeddings

    def get_positive_instance(self, in_batch=True):
        if in_batch:
            return [self.postive_n, self.postive_m, self.postive_f]
        else:
            return [self.batch_n[0:self.config.batchsize], self.batch_m[0:self.config.batchsize], self.batch_f[0:self.config.batchsize]]

    def get_positive_syn_instance(self, in_batch=True):
        if in_batch:
            return [self.postive_n, self.postive_m]
        else:
            return [self.batch_n[0:self.config.batchsize], self.batch_m[0:self.config.batchsize]]

    def get_negative_instance(self, in_batch=True):
        if in_batch:
            return [self.negative_n, self.negative_m, self.negative_f]
        else:
            return [self.batch_n[self.config.batchsize:self.config.batch_seq_size],
                    self.batch_m[self.config.batchsize:self.config.batch_seq_size],
                    self.batch_f[self.config.batchsize:self.config.batch_seq_size]]

    def get_negative_syn_instance(self, in_batch=True):
        if in_batch:
            return [self.negative_n, self.negative_m]
        else:
            return [self.batch_n[self.config.batchsize:self.config.batch_seq_size],
                    self.batch_m[self.config.batchsize:self.config.batch_seq_size]]

    def input_def(self):
        config = self.config
        self.batch_n = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_m = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_f = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_y = tf.placeholder(tf.float32, [config.batch_seq_size])
        self.postive_n = tf.transpose(tf.reshape(self.batch_n[0:config.batchsize], [1, -1]), [1, 0])
        self.postive_m = tf.transpose(tf.reshape(self.batch_m[0:config.batchsize], [1, -1]), [1, 0])
        self.postive_f = tf.transpose(tf.reshape(self.batch_f[0:config.batchsize], [1, -1]), [1, 0])
        self.negative_n = tf.transpose(tf.reshape(self.batch_n[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.negative_m = tf.transpose(tf.reshape(self.batch_m[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.negative_f = tf.transpose(tf.reshape(self.batch_f[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.parameter_lists = []

    def embedding_def(self):
        pass

    def pre_instance(self):
        pass

    def projection_loss(self):
        pass

    def __init__(self, config):
        self.config = config

        with tf.name_scope("input"):
            self.input_def()

        with tf.name_scope("embedding"):
            self.embedding_def()

        with tf.name_scope("instance"):
            self.pre_instance()

        with tf.name_scope("pro_loss"):
            self.projection_loss()
