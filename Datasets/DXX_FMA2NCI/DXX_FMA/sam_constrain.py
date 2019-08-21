import json

negitive_sampling_constrain = {}
sbpt = {}
dis = {}
f = open('./train2id.txt', 'r')
train2id_all = []
for line in f.readlines()[1:]:
    train2id_all.append(line)
f.close()
for line in train2id_all:
    train2id = line.strip('\t').split(" ")
    leftent, rightent, rel = int(train2id[0]), int(train2id[1]), int(train2id[2].replace('\n', ''))
    if rel == 0 or rel == 1:
        if str(leftent) not in sbpt.keys():
            sbpt[str(leftent)] = []
        else:
            pass
        if rightent not in sbpt[str(leftent)]:
            sbpt[str(leftent)].append(rightent)
        else:
            pass
        if str(rightent) not in sbpt.keys():
            sbpt[str(rightent)] = []
        else:
            pass
        if leftent not in sbpt[str(rightent)]:
            sbpt[str(rightent)].append(leftent)
        else:
            pass
    if rel == 2:
        if str(leftent) not in dis.keys():
            dis[str(leftent)] = []
        else:
            pass
        if rightent not in dis[str(leftent)]:
            dis[str(leftent)].append(rightent)
        else:
            pass
        if str(rightent) not in dis.keys():
            dis[str(rightent)] = []
        else:
            pass
        if leftent not in dis[str(rightent)]:
            dis[str(rightent)].append(leftent)
        else:
            pass
negitive_sampling_constrain['sbpt'] = sbpt
negitive_sampling_constrain['dis'] = dis

f = open('./neg_constrain.json', 'w')
json.dump(negitive_sampling_constrain, f)
f.close()