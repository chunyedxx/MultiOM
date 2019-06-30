import json
# 针对于训练矩阵的数据集，human_whole，和mouse_whole数据集，矩阵约束需要。
# 1.首先读入数据集，针对于数据集中的实体寻找在原来数据集中的父实体。
f2f_train_triple_path = "../Datasets/DXX/DXX_mouse_whole/f2f/train.txt"
f2s_train_triple_path = "../Datasets/DXX/DXX_mouse_whole/f2s/train.txt"
f2f_entity2id_path = "../Datasets/DXX/DXX_mouse_whole/f2f/entity2id.txt"
f2s_entity2id_path = "../Datasets/DXX/DXX_mouse_whole/f2s/entity2id.txt"
f2f_embedding_path = "../res/mouse_whole/f2f.embedding.vec.json"
f2s_embedding_path = "../res/mouse_whole/f2s.embedding.vec.json"

human_whole_train_path = "../Datasets/DXX/DXX_mouse_whole/train.txt"
human_train_path = "../Datasets/DXX/DXX_mouse/train.txt"
human_entity_path = "../Datasets/DXX/DXX_mouse/entity2id.txt"
human_embedding_path = "../res/mouse/mouse.embedding.vec.json"
whole_train_path = "../Datasets/DXX/DXX_whole/train.txt"
whole_entity_path = "../Datasets/DXX/DXX_whole/entity2id.txt"
whole_embedding_path = "../res/whole/whole.embedding.vec.json"
with open(human_whole_train_path, "r") as human_whole_train:
    human_whole_triple = list(human_whole_train)
with open(human_train_path, "r") as human_train:
    human_train_triple = list(human_train)
with open(human_entity_path, "r") as human_entity:
    human_entitys = list(human_entity)
with open(human_embedding_path, "r") as human_embedding:
    human_embeddings = json.load(human_embedding)["ent_embeddings"]
with open(whole_train_path, "r") as whole_train:
    whole_train_triple = list(whole_train)
with open(whole_entity_path, "r") as whole_entity:
    whole_entitys = list(whole_entity)
with open(whole_embedding_path, "r") as whole_embedding:
    whole_embeddings = json.load(whole_embedding)["ent_embeddings"]


with open(f2f_train_triple_path, "w") as f2f:
    with open(f2s_train_triple_path, "w") as f2s:
        i = 0
        j = 0
        for triple in human_whole_triple:
            substr = triple[triple.index("\t") + 1:triple.rindex("\t")]
            for human_tr_tri in human_train_triple:
                humans = human_tr_tri[:human_tr_tri.index("\t")]
                if (human_tr_tri[human_tr_tri.index("\t")+1:human_tr_tri.index("\t")+11] == "subClassof" and
                    triple[:triple.index("\t")] == humans):
                    triple_ent = substr[substr.index("\t") + 1:substr.rindex("\t")]
                    for whole_tr_tri in whole_train_triple:
                        substr_whole = whole_tr_tri[whole_tr_tri.index("\t") + 1:whole_tr_tri.rindex("\t") + 1]
                        humanf = human_tr_tri[human_tr_tri.rindex("\t") - 10:human_tr_tri.rindex("\t")]
                        wholef = substr_whole[substr_whole.index("\t") + 1:substr_whole.rindex("\t")]
                        wholes = whole_tr_tri[:whole_tr_tri.index("\t")]
                        if triple_ent == wholes:
                            f2f.write(humanf + "\tf2f\t" + wholef + "\t" + str(i) + "\n")
                            print(i)
                            i += 1

                        if triple_ent == wholef:
                            f2s.write(humanf + "\tf2s\t" + wholes + "\t" + str(j) + "\n")
                            print(j)
                            j += 1


with open(f2f_train_triple_path, "r") as f2f:
    f2f_triple = list(f2f)
with open(f2f_entity2id_path, "w") as f2f_entity:
    i = 0
    for triple in f2f_triple:
        leftent = triple[:triple.index("\t")]
        f2f_entity.write(leftent + "\t" + str(i) + "\n")
        i += 1
    j = i + 1
    for triple in f2f_triple:
        substr = triple[triple.index("\t") + 1:triple.rindex("\t") + 1]
        rightent = substr[substr.index("\t") + 1:substr.rindex("\t")]
        f2f_entity.write(rightent + "\t" + str(j) + "\n")
        j += 1

#
#
with open(f2s_train_triple_path, "r") as f2s:
    f2s_triple = list(f2s)
with open(f2s_entity2id_path, "w") as f2s_entity:
    i = 0
    for triple in f2s_triple:
        leftent = triple[:triple.index("\t")]
        f2s_entity.write(leftent + "\t" + str(i) + "\n")
        i += 1
    j = i + 1
    for triple in f2s_triple:
        substr = triple[triple.index("\t") + 1:triple.rindex("\t") + 1]
        rightent = substr[substr.index("\t") + 1:substr.rindex("\t")]
        f2s_entity.write(rightent + "\t" + str(j) + "\n")
        j += 1

with open(f2f_entity2id_path, "r") as f2f_entity:
    entity_list1 = list(f2f_entity)

f2f_embedding_list = []
f2f_embedding_dict = {}
for entity_ in entity_list1:
    entity = entity_[:entity_.index("\t")]
    # if "NCI_C" in entity:
    if "MA_00" in entity:
        for human_entity_ in human_entitys:
            human_entity = human_entity_[:human_entity_.index("-")]
            if entity == human_entity:
                human_entity_id = int(human_entity_[human_entity_.index("\t") + 1:human_entity_.index("\n")])
                f2f_embedding_list.append(human_embeddings[human_entity_id])
    else:
        for whole_entity_ in whole_entitys:
            whole_entity = whole_entity_[:whole_entity_.index("\t")]
            if entity == whole_entity:
                whole_entity_id = int(whole_entity_[whole_entity_.index("\t") + 1:whole_entity_.index("\n")])
                f2f_embedding_list.append(whole_embeddings[whole_entity_id])
with open(f2f_embedding_path, "w") as f2f_embedding:
    f2f_embedding_dict["ent_embeddings"] = f2f_embedding_list
    json.dump(f2f_embedding_dict, f2f_embedding)



with open(f2s_entity2id_path, "r") as f2s_entity:
    entity_list1 = list(f2s_entity)

f2s_embedding_list = []
f2s_embedding_dict = {}
for entity_ in entity_list1:
    entity = entity_[:entity_.index("\t")]
    # if "NCI_C" in entity:
    if "MA_00" in entity:
        for human_entity_ in human_entitys:
            human_entity = human_entity_[:human_entity_.index("-")]
            if entity == human_entity:
                human_entity_id = int(human_entity_[human_entity_.index("\t") + 1:human_entity_.index("\n")])
                f2s_embedding_list.append(human_embeddings[human_entity_id])
    else:
        for whole_entity_ in whole_entitys:
            whole_entity = whole_entity_[:whole_entity_.index("\t")]
            if entity == whole_entity:
                whole_entity_id = int(whole_entity_[whole_entity_.index("\t") + 1:whole_entity_.index("\n")])
                f2s_embedding_list.append(whole_embeddings[whole_entity_id])
with open(f2s_embedding_path, "w") as f2s_embedding:
    f2s_embedding_dict["ent_embeddings"] = f2s_embedding_list
    json.dump(f2s_embedding_dict, f2s_embedding)


