import xlrd
nci_entity_path = '..\Datasets\DXX\DXX_NCI\entity2id_completelyname.txt'  # 这是带概念名字的nci entity路径
ma_entity_path = '..\Datasets\DXX\DXX_MA\entity2id_xompletelyname.txt'   # 这是带概念名字的ma entity路径
fma_entity_path = '..\\Datasets\DXX\DXX_FMA\entity2id.txt'  # 这是带概念名字的fma entity路径
anatomy_path = '..\\anatomy.txt'     # 这是同义词词典的路径
nci_synonyms = '..\\nci_txt.txt'     # 这是nci同义词路径
ma_synonyms = '..\\ma_txt.txt'      # 这是ma同义词路径
train_txt_path = '..\\2.txt'


# 实体概念标准化，即参照anatomy,txt中的同义词，统一转化为--右边的格式
def entity_standardizing(entity_path):
    entity_list = []
    equivalence_dict = {}
    i = 1
    with open(anatomy_path, 'r') as f:
        for line in f:
            equivalence_ = line.strip().split('--')
            equivalence_dict[equivalence_[0]] = equivalence_[1]
    with open(entity_path, 'r') as f:
        for line in f:
            entity_id = line.strip().split('\t')
            if '--' in entity_id[0]:
                entity_name = entity_id[0][entity_id[0].index('-')+2:]
            else:
                entity_name = entity_id[0]
            if entity_name in equivalence_dict.keys() and '--' not in entity_id[0]:
                entity_list.append(entity_name + '--' + equivalence_dict[entity_name])
            elif entity_name in equivalence_dict.keys() and '--' in entity_id[0]:
                entity_numb = entity_id[0][:entity_id[0].index('-')+2]
                entity_list.append(entity_numb + equivalence_dict[entity_name])
            elif '--' in entity_id[0]:
                entity_list.append(entity_id[0])
            else:
                entity_list.append(entity_id[0] + '--' + entity_id[0])
    return entity_list


# 基于 tokenWord 的标准化
def tokeningword(string):
    son_string = ''
    for i in range(len(string)-1):
        if string[i].isupper() and i == 0:
            son_string += string[i]
        elif string[i].isupper() and string[i+1].islower() and i != 0:
            son_string += ' ' + string[i]
        elif string[i] == '-' or string[i] == '_' or string[i] == '.' or string[i] == ',':
            son_string += ' '
        elif string[i].isupper() and string[i+1].isupper():
            son_string += string[i]
        elif string[i].islower() and string[i+1].isupper():
            son_string += string[i] + ' '
        else:
            son_string += string[i]
    if len(string) > 0:
        son_string = son_string + string[-1]
        son_string = son_string.replace(" ", " ").strip()
    son_string = son_string.lower().replace('_|-', '')
    return son_string.replace('  ', ' ')


def entity_synonyms_dict(entity_path):
    entitys_syn_sict = {}
    syn_list = []
    if 'NCI' in entity_path or 'MA' in entity_path:
        if 'NCI' in entity_path:
            _synonyms = nci_synonyms
        else:
            _synonyms = ma_synonyms
        with open(_synonyms, 'r') as syn:
            for line in syn:
                syn_list.append(line)
    standardized_entity_list = entity_standardizing(entity_path)
    with open(entity_path, 'r') as f:
        for line in f:
            entity_ = line.strip().split('\t')[0]        # 这里我们只需要知道简写的概念的序号名字
            if '--' in entity_:
                entity_name = entity_[entity_.index('-')+2:]  # 名字
                entity_num = entity_[:entity_.index('-')]  # 序号名字
            else:
                entity_num = entity_
                entity_name = entity_
            entity_syn_list = [entity_name]
            for standardized_entity_ in standardized_entity_list:
                entity_std_name = standardized_entity_[standardized_entity_.index('-')+2:]
                entity_std = standardized_entity_[:standardized_entity_.index('-')]
                if entity_num == entity_std:
                    entity_syn_list.append(entity_std_name)
            entity_syn_list.append(tokeningword(entity_name))
            if 'NCI' in entity_path or 'MA' in entity_path:
                    for line in syn_list:
                        ent_syn = line.strip().split('--')
                        if entity_num == ent_syn[0]:
                            syn_son_list = list(ent_syn[2].strip().split(','))
                            if '' in syn_son_list:
                                syn_son_list.remove('')
                            for i in syn_son_list:
                                entity_syn_list.append(tokeningword(i))
            entitys_syn_sict[entity_num] = entity_syn_list
    return entitys_syn_sict


nci_entitys_syn_dict = entity_synonyms_dict(nci_entity_path)
print("nci_entitys_syn_dict is loaded")
ma_entitys_syn_dict = entity_synonyms_dict(ma_entity_path)
print("ma_entitys_syn_dict is loaded")
fma_entitys_syn_dict = entity_synonyms_dict(fma_entity_path)
print("fma_entitys_syn_dict is loaded")
i = 1
nci_fma_equivalence = []
ma_fma_equivalence = []
for item_fma in fma_entitys_syn_dict.items():
    print("item_fma is cycled to : %ld" % i)
    i += 1
    for item_nci in nci_entitys_syn_dict.items():
        for synnci in item_nci[1]:
            if synnci in item_fma[1]:
                nci_fma_equivalence.append((item_nci[0], item_fma[0]))
                break
    for item_ma in ma_entitys_syn_dict.items():
        for synma in item_ma[1]:
            if synma in item_fma[1]:
                ma_fma_equivalence.append((item_ma[0], item_fma[0]))
                break
print(len(nci_fma_equivalence))
print(len(ma_fma_equivalence))
with open(train_txt_path, 'w') as f:
    for nci_fma_ in nci_fma_equivalence:
        print(nci_fma_)
        for ma_fma_ in ma_fma_equivalence:
            if nci_fma_[1] == ma_fma_[1]:
                f.write(nci_fma_[0] + '\t' + ma_fma_[0] + '\t' + ma_fma_[1] + '\n')
# nci 与 ma 的实体对应
nci_ma_list = []
with open(train_txt_path, 'w') as f:
    for item_nci in nci_entitys_syn_dict.items():
        for item_ma in ma_entitys_syn_dict.items():
            for synma in item_ma[1]:
                if synma in item_nci[1] and item_nci[0] + '\t' + item_ma[0] not in nci_ma_list:
                    f.write(item_nci[0] + '\t' + item_ma[0] + '\n')
                    nci_ma_list.append(item_nci[0] + '\t' + item_ma[0])
                    break
