fma_entity_path = 'F:\py programmes\programmes\OpenKE\\benchmarks\DXX\DXX_FMA\entity2id.txt'
fma_train_path = 'F:\py programmes\programmes\OpenKE\\benchmarks\DXX\DXX_FMA\\train.txt'
writein_path = 'F:\py programmes\programmes\OpenKE\ontology_analyse.txt'
entity_list = []
train_list = []
with open(fma_entity_path, 'r') as entity:
    for ent_ in entity:
        entity_list.append(ent_[:ent_.index('\t')])
with open(fma_train_path, 'r') as train:
    for tra_ in train:
        train_list.append(tra_[:tra_.rindex('\t')])
with open(writein_path, 'w') as f:
    for ent in entity_list:
        lef_times = 0
        right_times = 0
        for tra_ in train_list:
            if ent == tra_[:tra_.index('\t')]:
                lef_times += 1
            tail = tra_[tra_.index('\t')+1:]
            if ent == tail[tail.index('\t')+1:]:
                right_times += 1
        print(ent + ':' + ' ' + '%ld' % lef_times)
        print(ent + ':' + ' ' + '%ld' % right_times)
        f.write(ent + ":" + str(lef_times) + "    " + str(right_times) + '\t\n')
