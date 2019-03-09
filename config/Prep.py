import numpy as np
import timeit
import os
from collections import defaultdict


class Prep(object):

    def __init__(self):
        self.negative_sampling = 'unif'
        self.triple_train = []
        self.ncientitytotal = 0
        self.maentitytotal = 0
        self.fmaentitytotal = 0
        self.ncient2id = {}
        self.tripletotal = 0
        self.in_path = "./benchmarks/DXX/DXX_UQU"
        self.bern = 0
        self.hidden_size = 100
        self.dimension = self.hidden_size
        self.nbatches = 0
        self.negative_ent = 1

    def get_batch(self):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)
        start = 0
        num = int(int(n_triple / self.nbatches))
        while start < n_triple:
            start_t = timeit.default_timer()
            end = min(start + num, n_triple)
            size = end - start
            train_triple_positive = list([self.triple_train[x] for x in rand_idx[start:end]])
            self.p_positive_batch_n = list([self.triple_train[x][0] for x in rand_idx[start:end]])
            self.p_positive_batch_m = list([self.triple_train[x][1] for x in rand_idx[start:end]])
            self.p_positive_batch_f = list([self.triple_train[x][2] for x in rand_idx[start:end]])
            self.p_batch_n = self.p_positive_batch_n
            self.p_batch_m = self.p_positive_batch_m
            self.p_batch_f = self.p_positive_batch_f
            start = end
            prepare_t = timeit.default_timer() - start_t
            yield self.p_batch_n, self.p_batch_m, self.p_batch_f, prepare_t

    def load_data(self):
        # if self.in_path != None:
            print('imported data path is: ' + self.in_path)
            with open(os.path.join(self.in_path, 'nci_entity2id.txt')) as f:
                self.ncient2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
                self.id2ncient = {value: key for key, value in self.ncient2id.items()}
            with open(os.path.join(self.in_path, 'ma_entity2id.txt')) as f:
                self.maent2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            with open(os.path.join(self.in_path, 'fma_entity2id.txt')) as f:
                self.fmaent2id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
            self.id2maent = {value: key for key, value in self.maent2id.items()}
            self.n_m_f = defaultdict(set)
            self.f_m_n = defaultdict(set)
            self.triple_train = self.load_triple()
            self.triple = self.triple_train
            self.maentitytotal = len(self.maent2id)
            self.ncientitytotal = len(self.ncient2id)
            self.fmaentitytotal = len(self.fmaent2id)
            self.tripletotal = len(self.triple_train)
            print('ncientity number: ' + str(self.ncientitytotal))
            print('maentity number: ' + str(self.maentitytotal))
            print('fmaentity number: ' + str(self.fmaentitytotal))
            print('training triple number: ' + str(self.tripletotal))
            if self.negative_sampling == 'bern':
                # {relation_id:[headid1, headid2,...]}
                self.maentity_property_head = {x: [] for x in range(self.maentitytotal)}
                # {relation_id:[tailid1, tailid2,...]}
                self.maentity_property_tail = {x: [] for x in range(self.maentitytotal)}
                for t in self.triple_train:
                    self.maentity_property_head[t[1]].append(t[0])
                    self.maentity_property_tail[t[1]].append(t[2])
                # {relation_id: p, ...} 0< num <1, and for relation replace head entity with the property p
                self.maentity_property = {x: (len(set(self.maentity_property_tail[x]))) /
                (len(set(self.maentity_property_head[x])) + len(set(self.maentity_property_tail[x])))
                                          for x in self.maentity_property_head.keys()}
            # else:
                # print("unif set do'n need to calculate hpt and tph")

    def load_triple(self):
        if self.in_path != None:
            triple_list = []  # [(head_id, relation_id, tail_id),...]
            with open(os.path.join(self.in_path, 'train.txt')) as f:
                for line in f.readlines():
                    line_list = line.strip().split('\t')
                    assert len(line_list) == 3
                    nciid = self.ncient2id[line_list[0]]
                    maid = self.maent2id[line_list[1]]
                    fmaid = self.fmaent2id[line_list[2]]
                    triple_list.append((nciid, maid, fmaid))
                    self.n_m_f[(nciid, maid)].add(fmaid)
                    self.f_m_n[(fmaid, maid)].add(nciid)
            return triple_list
        else:
            print("the file path is not a correctly directory")

    def set_batches(self, nbatches):
        self.nbatches = nbatches

    def set_in_path(self, path):
        self.in_path = path

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_bern(self, bern):
        self.bern = bern

