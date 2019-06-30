train_path = '..\Datasets\DXX\DXX_FMA\\train.txt'
dis_writein_path = '..\\train.txt'

f = open(train_path, 'r')
train_dict = {}
list_ = list(f)
f.close()
for line in list_:
    triple = line.strip().split('\t')
    if triple[2] not in train_dict.keys():
        train_dict[triple[2]] = []
        train_dict[triple[2]].append(triple[0])
    elif triple[0] not in train_dict[triple[2]]:
        train_dict[triple[2]].append(triple[0])
    else: pass
f = open(dis_writein_path, 'w')
k = 78985
dis_list = []
for value_list in train_dict.values():
    if len(value_list) > 1:
        for i in range(len(value_list)):
            j = i + 1
            while j < len(value_list):
                if (value_list[i], value_list[j]) not in dis_list and (value_list[j], value_list[i]) not in dis_list:
                    dis_list.append((value_list[i], value_list[j]))
                    f.write(value_list[i] + '\t' + 'disjointwith' + '\t' + value_list[j] + '\t' + str(k) + '\n')
                    print(value_list[i] + '\t' + 'disjointwith' + '\t' + value_list[j] + '\t' + str(k))
                    k += 1
                else: pass
                j += 1
f.close()
