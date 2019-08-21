import numpy as np
from TfidfSimility import idf_similarity, cos_distance
from StableMarriage import stable_marriage
import json


def entity_idname_list(path):
    ent_list = []
    f = open(path, "r")
    for line in f.readlines():
        nent_name = line[:10]
        ent_list.append(nent_name)
    f.close()
    return ent_list


f = open('..\\reference\\referencemap.txt', "r")
referencemap = list(f)
f.close()


def get_embed_and_proj_mat():
    f = open('..\\res\\uqu06123.embedding.vec.json', "r")
    parameters_dict = json.load(f)
    f.close()
    h_uqu_vec_list = parameters_dict["nci_ent_embeddings"][:3298]
    m_uqu_vec_list = parameters_dict["ma_ent_embeddings"][:2737]
    h_w_matric = parameters_dict["n2f_transfer_matrix"]
    m_w_matric = parameters_dict["m2f_transfer_matrix"]
    h_w_matric = np.reshape(h_w_matric, [50, 50])
    m_w_matric = np.reshape(m_w_matric, [50, 50])
    return h_uqu_vec_list, m_uqu_vec_list, h_w_matric, m_w_matric


h_uqu_vec_list, m_uqu_vec_list, h_w_matric, m_w_matric = get_embed_and_proj_mat()


def align_values_dict_fun(ma_list, nci_list, threshold=0.95):
    i = 1
    trained_align_values_dict = {}
    threshold_val_alignments = []
    for maent in ma_list:
        ma_uqu_vec = m_uqu_vec_list[ma_list.index(maent)]
        for ncient in nci_list:
            nci_uqu_vec = h_uqu_vec_list[nci_list.index(ncient)]
            simility_trained = cos_distance(np.dot(h_w_matric, np.transpose(nci_uqu_vec)),
                                        np.dot(m_w_matric, np.transpose(ma_uqu_vec)))
            trained_align_values_dict[maent + '\t' + ncient] = simility_trained
            if simility_trained >= threshold: threshold_val_alignments.append((maent, ncient))
            print(i)
            i += 1
    return trained_align_values_dict, threshold_val_alignments


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


ma_list = entity_idname_list("..\datasets\DXX\DXX_MA\entity2id_completelyname.txt")
nci_list = entity_idname_list("..\datasets\DXX\DXX_NCI\entity2id_completelyname.txt")
threshold = 0.95  # 0.95时会降低4个匹配
trained_align_values_dict, threshold_val_alignments = align_values_dict_fun(ma_list, nci_list, threshold)
ma2nci = total_sub_dict(ma_list, nci_list, trained_align_values_dict)
nci2ma = total_sub_dict(nci_list, ma_list, trained_align_values_dict)
total_dict_1 = {'ma2nci': ma2nci, 'nci2ma': nci2ma}
one2onealign = stable_marriage(total_dict_1)


def alignment_filter(ma, nci, ma2nci, align_values_dict):
    alignments = []
    simility_ = align_values_dict[ma + '\t' + nci]
    if ma2nci[ma].index(nci) <= 1:
        max_nci = ma2nci[ma][0]
        alignments.append((ma, max_nci))
        if align_values_dict[ma + '\t' + max_nci] - simility_ <= 0.0005 and (ma, nci) not in alignments:
            # 只有这里是0.005，其他地方是0.00005
            alignments.append((ma, nci))
        else: pass
    else: pass
    return alignments


def alignments_match(threshold_val_alignments):
    alignments = []
    for align in threshold_val_alignments:
        ma, nci = align[0], align[1]
        aligns = alignment_filter(ma, nci, ma2nci, trained_align_values_dict)
        for couple in aligns:
            if couple not in alignments: alignments.append(couple)
    return alignments


alignments = alignments_match(threshold_val_alignments)
print("改进后的匹配算法")
corres = 0
with open('..res\\align_onto.txt', 'w') as f:
    for i in alignments:
        if i[0] + ',' + i[1] + ',=\n' in referencemap:
            corres += 1
            f.write(i[0] + ',' + i[1] + ',=\n')
print('**************** onto ****************')
print('匹配总数：%ld' % len(alignments))
print("正确个数：%ld" % corres)
pre = corres / len(alignments)
rec = corres / len(referencemap)
F = pre * rec * 2 / (pre + rec)
print("pre :%f \t rec:%f \t F:%f" % (pre, rec, F))
