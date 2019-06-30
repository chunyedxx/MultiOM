import numpy as np
import random
import timeit
import json
import os


class Prep(object):

    def get_batch(self):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)
        start = 0
        batchsize = int(n_triple / self.nbatches)
        while start < batchsize * self.nbatches:
            start_t = timeit.default_timer()
            end = min(start + batchsize, n_triple)
            size = end - start
            train_triple_positive = list([self.triple_train[x] for x in rand_idx[start:end]])
            train_triple_negative = []
            for t in train_triple_positive:
                random_num = np.random.random()
                nci_entity_id_list = list(range(self.ncientitytotal))
                ma_entity_id_list = list(range(self.maentitytotal))
                nci_entity_id_list.remove(t[0])
                ma_entity_id_list.remove(t[1])
                if str(t[0]) in self.nci_constrain_dict['dis'].keys():
                    dis_samp_nci = self.nci_constrain_dict['dis'][str(t[0])]
                else: dis_samp_nci = []
                if str(t[1]) in self.ma_constrain_dict['dis'].keys():
                    dis_samp_ma = self.ma_constrain_dict['dis'][str(t[1])]
                else: dis_samp_ma = []
                if self.negative_ent <= len(dis_samp_nci):
                    replace_nci_entity_id_list = random.sample(dis_samp_nci, self.negative_ent)
                else:
                    for ntri in self.nci_constrain_dict['sbpt'][str(t[0])] + dis_samp_nci:
                        if ntri in nci_entity_id_list: nci_entity_id_list.remove(ntri)
                    tem = random.sample(nci_entity_id_list, self.negative_ent - len(dis_samp_nci))
                    replace_nci_entity_id_list = dis_samp_nci + tem
                if self.negative_ent <= len(dis_samp_ma):
                    replace_ma_entity_id_list = random.sample(dis_samp_ma, self.negative_ent)
                else:
                    for ntri in self.ma_constrain_dict['sbpt'][str(t[1])] + dis_samp_ma:
                        if ntri in ma_entity_id_list: ma_entity_id_list.remove(ntri)
                    tem = random.sample(ma_entity_id_list, self.negative_ent - len(dis_samp_ma))
                    replace_ma_entity_id_list = dis_samp_ma + tem
                if self.negative_sampling == 'unif':
                    replace_nci_probability = 0.5
                elif self.negative_sampling == 'bern':
                    replace_nci_probability = self.fmaentity_property[t[2]]
                else:
                    raise NotImplementedError("Dose not support %s negative_sampling" % self.negative_sampling)
                if self.modelname != "ontomap":
                    for mt in replace_ma_entity_id_list:
                        train_triple_negative.append((t[0], mt))
                else:
                    if random_num < replace_nci_probability:
                        for nt in replace_nci_entity_id_list:
                            train_triple_negative.append((nt, t[1], t[2]))
                    else:
                        for mt in replace_ma_entity_id_list:
                            train_triple_negative.append((t[0], mt, t[2]))
                self.p_positive_batch_n = list([x[0] for x in train_triple_positive])
                self.p_positive_batch_m = list([x[1] for x in train_triple_positive])
                self.p_negative_batch_n = list([triple[0] for triple in train_triple_negative])
                self.p_negative_batch_m = list([triple[1] for triple in train_triple_negative])
                self.p_batch_n = self.p_positive_batch_n + self.p_negative_batch_n
                self.p_batch_m = self.p_positive_batch_m + self.p_negative_batch_m
                if self.modelname == "ontomap":
                    self.p_positive_batch_f = list([x[2] for x in train_triple_positive])
                    self.p_negative_batch_f = list([triple[2] for triple in train_triple_negative])
                    self.p_batch_f = self.p_positive_batch_f + self.p_negative_batch_f
            start = end
            prepare_t = timeit.default_timer() - start_t
            if self.modelname != "ontomap":
                yield self.p_batch_n, self.p_batch_m, prepare_t
            else:
                yield self.p_batch_n, self.p_batch_m, self.p_batch_f, prepare_t

    def __init__(self):
        self.negative_sampling = 'unif'
        self.triple_train = []
        self.ncientitytotal = 0
        self.maentitytotal = 0
        self.fmaentitytotal = 0
        self.ncient2id = {}
        self.tripletotal = 0
        self.constrain_nci_tripletotal = 0
        self.constrain_ma_tripletotal = 0
        self.in_path = "./Datasets/DXX/DXX_UQU"
        self.nbatches = 0
        self.negative_ent = 1
        self.modelname = None

    def load_data(self):
        print('imported data path is: ' + self.in_path)
        with open(os.path.join(self.in_path, 'ncientity2id.txt')) as f:
            self.ncient2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        with open(os.path.join(self.in_path, 'maentity2id.txt')) as f:
            self.maent2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        with open(os.path.join(self.in_path[:-8], 'DXX_NCI\\neg_constrain.json')) as f:
            self.nci_constrain_dict = json.load(f)
        with open(os.path.join(self.in_path[:-8], 'DXX_MA\\neg_constrain.json')) as f:
            self.ma_constrain_dict = json.load(f)
        if self.modelname == 'ontomap':
            with open(os.path.join(self.in_path, 'fmaentity2id.txt')) as f:
                self.fmaent2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            self.fmaentitytotal = len(self.fmaent2id)
            print('fmaentity number: ' + str(self.fmaentitytotal))
        self.triple_train = self.load_triple('train.txt')
        self.ncientitytotal = len(self.ncient2id)
        self.maentitytotal = len(self.maent2id)
        self.tripletotal = len(self.triple_train)
        print('ncientity number: ' + str(self.ncientitytotal))
        print('maentity number: ' + str(self.maentitytotal))
        print('training triple number:' + str(self.tripletotal))
        if self.negative_sampling == 'bern':
            self.fmaentity_property_nci = {x: [] for x in range(self.fmaentitytotal)}
            self.fmaentity_property_ma = {x: [] for x in range(self.fmaentitytotal)}
            for t in self.triple_train:
                self.fmaentity_property_nci[t[2]].append(t[0])
                self.fmaentity_property_ma[t[2]].append(t[1])
            self.fmaentity_property = {x: (len(set(self.fmaentity_property_ma[x]))) / (len(set(self.fmaentity_property_nci[x])) + len(set(self.fmaentity_property_ma[x]))) for x in self.fmaentity_property_nci.keys()}

    def load_triple(self, filename):
        triple_list = []
        with open(os.path.join(self.in_path, filename)) as f:
            for line in f.readlines():
                train_list = line.strip().split('\t')
                if self.modelname != "ontomap":
                    nciid = self.ncient2id[train_list[0]]  # nci
                    maid = self.maent2id[train_list[1]]  # nci
                else:
                    nciid = self.ncient2id[train_list[0]]  # nci
                    maid = self.maent2id[train_list[1]]  # nci
                    fmaid = self.fmaent2id[train_list[2]]  # rel
                if self.modelname == "ontomap_syn":
                    triple_list.append((nciid, maid))
                else:
                    triple_list.append((nciid, maid, fmaid))
        return triple_list

    def set_batches(self, nbatches):
        self.nbatches = nbatches

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, path):
        self.in_path = path

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_negative_sampling(self, negative_sampling):
        self.negative_sampling = negative_sampling

    def model_name(self, name):
        self.modelname = name
