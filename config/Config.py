# coding:utf-8
import numpy as np
import tensorflow as tf
import json
from .Prep import Prep


class Config(Prep):

    def __init__(self):
        Prep.__init__(self)
        self.out_path = None
        self.train_times = 0
        self.alpha = 0.001
        self.log_on = 1
        self.dimension = 100
        self.exportName = None
        self.importName = None
        self.export_steps = 0
        self.opt_method = "SGD"
        self.optimizer = None


    def init(self):
        if self.in_path != None:
            self.load_data()
            self.ncienttotal = self.ncientitytotal
            self.maenttotal = self.maentitytotal
            self.fmaenttotal = self.fmaentitytotal
            self.batchsize = int(self.tripletotal / self.nbatches)
            self.batch_seq_size = self.batchsize * (1 + self.negative_ent)
            self.batch_n = np.zeros(self.batchsize * (1 + self.negative_ent), dtype=np.int64)
            self.batch_m = np.zeros(self.batchsize * (1 + self.negative_ent), dtype=np.int64)
            self.batch_f = np.zeros(self.batchsize * (1 + self.negative_ent), dtype=np.int64)

    def get_ent_total(self):
        return self.ncienttotal, self.maenttotal,  self.fmaenttotal

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_opt_method(self, method):
        self.opt_method = method

    def set_log_on(self, flag):
        self.log_on = flag

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_out_files(self, path):
        self.out_path = path

    def set_ent_dimension(self, dim):
        self.dimension = dim

    def set_train_times(self, times):
        self.train_times = times

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path, steps=0):
        self.exportName = path
        self.export_steps = steps

    def save_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, self.exportName)

    def restore_tensorflow(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, self.importName)

    def get_parameters_by_name(self, var_name):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    return self.sess.run(self.trainModel.parameter_lists[var_name])
                else:
                    return None

    def get_parameters(self, mode="numpy"):
        res = {}
        lists = self.trainModel.parameter_lists
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = self.get_parameters_by_name(var_name)
            else:
                res[var_name] = self.get_parameters_by_name(var_name).tolist()
        return res

    def save_parameters(self, path=None):
        if path == None:
            path = self.out_path
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def set_model(self, model):
        self.model = model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform=True)
                with tf.variable_scope("model", reuse=None, initializer=initializer):
                    self.trainModel = self.model(config=self)
                    if self.optimizer != None: pass
                    elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
                        self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.alpha,initial_accumulator_value=1e-20)
                    elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
                        self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
                    elif self.opt_method == "Adam" or self.opt_method == "adam":
                        self.optimizer = tf.train.AdamOptimizer(self.alpha)
                    else: self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
                    grads_and_vars = self.optimizer.compute_gradients(self.trainModel.pro_loss)
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)
                self.saver = tf.train.Saver()
                self.sess.run(tf.initialize_all_variables())

    def train_step(self,batch_h, batch_t, batch_r):
        feed_dict = {
            self.trainModel.batch_n: batch_h,
            self.trainModel.batch_m: batch_t,
            self.trainModel.batch_f: batch_r}
        _,pro_loss = self.sess.run([self.train_op, self.trainModel.pro_loss], feed_dict)
        return pro_loss

    def train_syn_step(self,batch_h, batch_t):
        feed_dict = {
            self.trainModel.batch_n: batch_h,
            self.trainModel.batch_m: batch_t}
        _,pro_loss = self.sess.run([self.train_op, self.trainModel.pro_loss], feed_dict)
        return pro_loss

    def run(self):
        with self.graph.as_default():
            with self.sess.as_default():
                if self.importName != None:
                    self.restore_tensorflow()
                for times in range(self.train_times):
                    pro_res = 0.0
                    if self.modelname == 'ontomapsyn':
                        for bn, bm, _ in self.get_batch():
                            pro = self.train_syn_step(bn, bm)
                            pro_res += pro
                        if self.log_on:
                            print("training times:%ld" % times)
                            print("synonyms loss:%f" % pro_res)
                            print("--------------------------")
                    else:
                        for bn, bm, bf, _ in self.get_batch():
                            pro = self.train_step(bn, bm, bf)
                            pro_res += pro
                        if self.log_on:
                            print("training times:%ld" % times)
                            print("projection loss:%f" % pro_res)
                            print("--------------------------")
                    if self.exportName != None and (self.export_steps != 0 and times % self.export_steps == 0):
                        self.save_tensorflow()
                if self.exportName != None:
                    self.save_tensorflow()
                if self.out_path != None:
                    self.save_parameters(self.out_path)
