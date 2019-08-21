import numpy as np
from tfidf_simility import idf_similarity, cos_distance
from stable_marriage import stable_marriage
import json


def entity_idname_list(path):
    ent_list = []
    f = open(path, "r")
    for line in f.readlines():
        nent_name = line[:10]
        ent_list.append(nent_name)
    f.close()
    return ent_list


def align_values_dict_fun(ma_list, nci_list):
    i = 1
    tfidf_align_values_dict = {}
    for maent in ma_list:
        for ncient in nci_list:
            simility_idf = idf_similarity(maent, ncient)
            tfidf_align_values_dict[maent + '\t' + ncient] = simility_idf
            print(i)
            i += 1
    return tfidf_align_values_dict


def total_sub_dict(list1, list2, align_values_dict):
    i = 1
    ent12ent2 = {}
    for ent1 in list1:
        ent1_dict = {}
        for ent2 in list2:
            if ent1 + '\t' + ent2 in align_values_dict.keys():
                ent1_dict[ent2] = align_values_dict[ent1 + '\t' + ent2]
            else: ent1_dict[ent2] = align_values_dict[ent2 + '\t' + ent1]
        sort_maent_dict = dict(sorted(ent1_dict.items(), key=lambda item: item[1], reverse=True))
        keys1 = list(sort_maent_dict.keys())
        ent12ent2[ent1] = keys1
        print("macycle:", i)
        i += 1
    return ent12ent2


ma_list = entity_idname_list("..\Datasets\DXX\DXX_MA\entity2id_completelyname.txt")
nci_list = entity_idname_list("..\Datasets\DXX\DXX_NCI\entity2id_completelyname.txt")
tfidf_align_values_dict = align_values_dict_fun(ma_list, nci_list)
ma2nci = total_sub_dict(ma_list, nci_list, tfidf_align_values_dict)
nci2ma = total_sub_dict(nci_list, ma_list, tfidf_align_values_dict)
total_dict_1 = {'ma2nci': ma2nci, 'nci2ma': nci2ma}
one2onealign = stable_marriage(total_dict_1)


f = open('..\\reference\\referencemap.txt', "r")
referencemap = list(f)
f.close()

corres = 0
align = []
for i,j in one2onealign.items():
    if tfidf_align_values_dict[i + '\t' + j] >= 0.8:
        align.append(i + ',' + j + ',=\n')
with open('..res\\align_tfidf.txt', 'w') as f:
    for i in align:
        if i in referencemap:
            corres += 1
            tmp = i.split(',')
            ma, nci = tmp[0], tmp[1]
            f.write(i)
            # f.write(i.replace('\n', '') + str(tfidf_align_values_dict[ma + '\t' + nci]) + '\n')

print('**************** tfidf ****************')
print('匹配总数：%ld' % len(align))
print("正确个数：%ld" % corres)
pre = corres / len(align)
rec = corres / len(referencemap)
F = pre * rec * 2 / (pre + rec)
print("pre :%f \t rec:%f \t F:%f" % (pre, rec, F))
