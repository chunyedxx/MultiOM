import numpy as np
import json
from itertools import chain
import math as m


def cos_distance(x, y):
    vec_inner_product = np.dot(x, y)
    x_morm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    cos_dis = vec_inner_product / (x_morm * y_norm)
    return cos_dis


def idf_statistical():
    f = open("..\Datasets\DXX\DXX_MA\entity2id_completelyname.txt", 'r')
    malsit = list(f)
    f.close()
    f = open("..\Datasets\DXX\DXX_NCI\entity2id_completelyname.txt", 'r')
    ncilist = list(f)
    f.close()
    f = open('..\synonyms\\rep.json', 'r')
    rep = json.load(f)
    concept_list = malsit + ncilist
    concept_nums = len(concept_list)
    idf_dict = {}
    name_concept_dict = {}
    for line in concept_list:
        name_concept = line.strip().split('\t')[0]
        name = name_concept[:10]
        concept = name_concept[12:].lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')\
                                    .replace('(', '').replace(')', "").replace("'s", '')
        for key, val in rep.items():
            for i in val:
                if i in concept:concept = concept.replace(i, key)
                else: pass
        concept_tokens = concept.split(' ')
        name_concept_dict[name] = concept
        for token in concept_tokens:
            if token not in idf_dict.keys(): idf_dict[token] = 1
            else: idf_dict[token] += 1
    for token in idf_dict.keys(): idf_dict[token] = m.log(concept_nums / idf_dict[token])
    return name_concept_dict, idf_dict


def token_vec(idf_dict):
    vec_dict = {}
    f = open("..\pretrained_vectors\pretrained-wikipedia-pubmed-and-PMC-w2v.txt", 'r')
    for line in f:
        vec_list = []
        a = line.split(' ')
        name, vec = a[0], a[1:]
        for i in vec: vec_list.append(float(i))
        if name not in vec_dict.keys(): vec_dict[name] = np.reshape(vec_list, newshape=[1, 200])
    f.close()
    for token in idf_dict.keys():
        if token not in vec_dict.keys():
            vec_dict[token] = np.random.randint(low=1, high=5, size=[1, 200]) * 0.1
        else: pass
    return vec_dict


name_concept_dict, idf_dict = idf_statistical()
token_vec_dict = token_vec(idf_dict)


def idf_similarity(maent, ncient):
    maname = name_concept_dict[maent]
    nciname = name_concept_dict[ncient]
    dict_sim = {}
    masim = ncisim = 0.0
    for matoken in maname.split(' '):
        mavec = token_vec_dict[matoken]
        for ncitoken in nciname.split(' '):
            ncivec = token_vec_dict[ncitoken]
            if matoken + '_' + ncitoken not in dict_sim.keys():
                dict_sim[matoken + '_' + ncitoken] = cos_distance(mavec, ncivec.T)
            else: pass
    madenominator = sum(idf_dict[i] for i in maname.split(' '))
    for matoken in maname.split(' '):
        tem_list = []
        for mancitoken in dict_sim.keys():
            if matoken == mancitoken[:mancitoken.index('_')]:
                # a = list(chain.from_iterable(dict_sim[mancitoken]))[0]
                a = dict_sim[mancitoken][0][0]
                tem_list.append(a)
        tem_list = sorted(tem_list)
        masim += tem_list[-1] * idf_dict[matoken] / madenominator
    ncidenominator = sum(idf_dict[i] for i in nciname.split(' '))
    for ncitoken in nciname.split(' '):
        tem_list_ = []
        for ncimatoken in dict_sim.keys():
            if ncitoken == ncimatoken[ncimatoken.index('_')+1:]:
                # a = list(chain.from_iterable(dict_sim[ncimatoken]))[0]
                a = dict_sim[ncimatoken][0][0]
                tem_list_.append(a)
        ncisim += sorted(tem_list_)[-1] * idf_dict[ncitoken] / ncidenominator
    return min(masim, ncisim)


def idf_similarity_str(maent, ncient):
    maname = name_concept_dict[maent]
    nciname = name_concept_dict[ncient]
    dict_sim = {}
    masim = ncisim = 0.0
    for matoken in maname.split(' '):
        for ncitoken in nciname.split(' '):
            if matoken + '_' + ncitoken not in dict_sim.keys():
                if matoken == ncitoken: dict_sim[matoken + '_' + ncitoken] = 1.0
                else: dict_sim[matoken + '_' + ncitoken] = 0.0
            else: pass
    madenominator = sum(idf_dict[i] for i in maname.split(' '))
    for matoken in maname.split(' '):
        tem_list = []
        for mancitoken in dict_sim.keys():
            if matoken == mancitoken[:mancitoken.index('_')]:
                a = dict_sim[mancitoken]
                tem_list.append(a)
        tem_list = sorted(tem_list)
        masim += tem_list[-1] * idf_dict[matoken] / madenominator
    ncidenominator = sum(idf_dict[i] for i in nciname.split(' '))
    for ncitoken in nciname.split(' '):
        tem_list_ = []
        for ncimatoken in dict_sim.keys():
            if ncitoken == ncimatoken[ncimatoken.index('_')+1:]:
                # a = list(chain.from_iterable(dict_sim[ncimatoken]))[0]
                a = dict_sim[ncimatoken]
                tem_list_.append(a)
        ncisim += sorted(tem_list_)[-1] * idf_dict[ncitoken] / ncidenominator
    return min(masim, ncisim)