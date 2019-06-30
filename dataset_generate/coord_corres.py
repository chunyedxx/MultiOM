
def mouse_whole_dataset():
    train2id = "..\\Datasets\DXX\DXX_MA\\train2id_all.txt"
    train = "..\\Datasets\DXX\DXX_MA\\train.txt"
    entity = "..\Datasets\DXX\DXX_MA\\ma_entity2id.txt"
    relation = '..\\Datasets\DXX\DXX_MA\\relation2id.txt'
    f = open(train, "r")
    train_triple = []
    for line in f:
        medium = line.strip().split('\t')
        train_triple.append((medium[0], medium[1], medium[2]))
    f.close()
    f = open(entity, "r")
    entity_id = []
    for line in f:
        medium = line.strip().split('\t')
        entity_id.append((medium[0], medium[1]))
    f.close()
    f = open(relation, "r")
    relation_id = []
    for line in f:
        medium = line.strip().split('\t')
        relation_id.append((medium[0], medium[1]))
    f.close()
    f = open(train2id, "w")
    i = 0
    for triple in train_triple:
        triple_list = []
        head = triple[0]
        rel = triple[1]
        tail = triple[2]
        for ent2id in entity_id:
            # if head == ent2id[0][:ent2id[0].index('-')]:
            if head == ent2id[0]:
                triple_list.append(ent2id[1])
        for ent2id in entity_id:
            # if tail == ent2id[0][:ent2id[0].index('-')]:
            if tail == ent2id[0]:
                triple_list.append(ent2id[1])
        for rel2id in relation_id:
            if rel == rel2id[0]:
                triple_list.append(rel2id[1])
        print(triple_list)
        f.write(triple_list[0] + ' ' + triple_list[1] + ' ' + triple_list[2] + '\n')
        i += 1
        print("*************************" + str(i))


a = mouse_whole_dataset()