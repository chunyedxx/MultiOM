# coding:utf-8
import numpy as np
import tensorflow as tf
import json


class Model(object):

    def get_config(self):
        return self.config

    def get_positive_instance(self, in_batch=True):
        if in_batch:
            return [self.postive_n, self.postive_m, self.postive_f]
        else:
            return [self.batch_n[0:self.config.batch_size], self.batch_m[0:self.config.batch_size], self.batch_f[0:self.config.batch_size]]

    def get_all_instance(self, in_batch=False):
        if in_batch:
            return [tf.transpose(tf.reshape(self.batch_n, [1 + self.config.negative_ent, -1]), [1, 0]),
                    tf.transpose(tf.reshape(self.batch_m, [1 + self.config.negative_ent, -1]), [1, 0]),
                    tf.transpose(tf.reshape(self.batch_f, [1 + self.config.negative_ent, -1]), [1, 0])]
        else:
            return [self.batch_n, self.batch_m, self.batch_f]

    def get_all_labels(self, in_batch=False):
        if in_batch:
            return [tf.transpose(tf.reshape(self.batch_y, [1 + self.config.negative_ent, -1]), [1, 0])]
        else:
            return [self.batch_y]


    def input_def(self):
        config = self.config
        self.batch_n = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_m = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_f = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_y = tf.placeholder(tf.float32, [config.batch_seq_size])
        size = int(config.batch_size)
        self.postive_n = tf.transpose(tf.reshape(self.batch_n[0:size], [1, -1]), [1, 0])
        self.postive_m = tf.transpose(tf.reshape(self.batch_m[0:size], [1, -1]), [1, 0])
        self.postive_f = tf.transpose(tf.reshape(self.batch_f[0:size], [1, -1]), [1, 0])
        self.parameter_lists = []

    def embedding_def(self):
        pass

    def pre_instance(self):
        pass

    def projection_loss(self):
        pass

    def triple_loss(self):
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

        with tf.name_scope("tri_loss"):
            self.triple_loss()
