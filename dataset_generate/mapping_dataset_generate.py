import json


# 第一步，制作数据集，针对DXX_ent_corres中的数据，首先将每一个txt文件中的实体对应到一列，因为我们需要做成这个框架可以识别的数据集，
# 并且id可以当前数据中排序即可.
def human_whole(readinpath, writeinpath):
    '''
    :param readinpath:待读入三元组txt文件的路径
    :param writeinpath: 待写入实体的txt文件路径
    :return: 无返回值，但会写入文件
    '''
    entitylist = []
    headlist = []
    taillist = []
    with open(readinpath) as readin:
        for line in readin.readlines():
            head = line[:line.index("\t")]
            bridge = line[line.index("\t")+1:line.rindex('\t')]
            tail = bridge[bridge.index("\t")+1:bridge.rindex("\t")]
            print(tail)
            if head not in headlist:
                headlist.append(head)
            if tail not in taillist:
                taillist.append(tail)
    entitylist.extend(headlist)
    entitylist.extend(taillist)
    i =0
    with open(writeinpath, 'w') as writein:
        for ele in entitylist:
            writein.write(ele + " " + str(i) + "\n")
            i += 1
    return "beautiful"


# 第二步，实体与坐标对应
def lookup_coordinate(transferentitypath, humanentitypath, wholeentitypath, preembeddingpath, writeinjson):
    '''
    :param transferentitypath: 这是用来训练投影矩阵的实体集合的路径
    :param entitypath: 这是原来三个本体中实体的路径
    :param embeddingpath: 这是预训练得到的向量的路径
    :param writeinjson: 这是匹配好id之后的待写入的json文件路径
    :return: 无返回值 如果一定要返回，那就指定“pretty”
    '''
    transferentitypath = "/home/user/ontomap_2/Datasets/DXX/DXX_human_whole/entity2id.txt"
    humanentitypath = "/home/user/ontomap_2/Datasets/DXX/DXX_human/entity2id.txt"
    preembeddingpath1 = "../res/human/human.embedding.vec.json"
    writeinjson = "../pipei_human.json"
    transembedding = []
    wirteinjson = {}
    entori1 = []
    entori2 = []
    with open(humanentitypath, "r") as entity1:
        for entorigin1 in entity1.readlines():
            ent1 = entorigin1[:entorigin1.index("-")]
            entori1.append(ent1)
    with open(preembeddingpath1, "r") as embedding1:
        embeddingdict1 = json.load(embedding1)                    # 预训练得到的向量

    with open(transferentitypath, "r") as transferentity:
        for transferent in transferentity.readlines():
            tra_ent = transferent[:transferent.index(" ")]
            if tra_ent in entori1:
                id = int(entori1.index(tra_ent))
                transembedding.append(embeddingdict1['ent_embeddings'][id])
    with open(writeinjson, 'w') as writein:
        wirteinjson["ent_embeddings"] = transembedding
        json.dump(wirteinjson, writein)

    return "pretty"


transferentitypath = ""
humanentitypath = ""
wholeentitypath = ""
preembeddingpath = ""
writeinjson = ""
a = lookup_coordinate(transferentitypath, humanentitypath, wholeentitypath, preembeddingpath, writeinjson)
