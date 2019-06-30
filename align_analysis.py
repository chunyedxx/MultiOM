
# 读入onto
with open('..\\align_onto.txt', 'r') as f:
    onto_list = []
    for line in f:
        tmp = line.replace('\n', '')
        if tmp not in onto_list:
            onto_list.append(tmp)
print('onto totally: %ld' % len(onto_list))
# 读入onto_syn
with open('..\\align_ontosyn.txt', 'r') as f:
    ontosyn_list = []
    for line in f:
        tmp = line.replace('\n', '')
        if tmp not in ontosyn_list:
            ontosyn_list.append(tmp)
print('ontosyn totally: %ld' % len(ontosyn_list))
# 读入tfidf
with open('..\\align_tfidf.txt', 'r') as f:
    tfidf_list = []
    for line in f:
        tmp = line.replace('\n', '')
        if tmp not in tfidf_list:
            tfidf_list.append(tmp)
print('tfidf totally: %ld' % len(tfidf_list))
# *******************tfidf，对比onto****************************
tfidf_onto_rest = []
for tf1 in tfidf_list:
    if tf1 not in onto_list:
        tfidf_onto_rest.append(tf1)
print('tfidf not in onto totally: %ld' % len(tfidf_onto_rest))
# *******************tfidf，对比ontosyn****************************
tfidf_ontosyn_rest = []
for tf2 in tfidf_list:
    if tf2 not in ontosyn_list:
        tfidf_ontosyn_rest.append(tf2)
print('tfidf not in ontosyn totally: %ld' % len(tfidf_ontosyn_rest))
# ************tfidf, 对比 合并onto + ontosyn**********************
onto_ontosyn_list = ontosyn_list[:]
for onto in onto_list:
    if onto not in onto_ontosyn_list:
        onto_ontosyn_list.append(onto)
tfidf_ontoontosyn_rest = []
for tf in tfidf_list:
    if tf not in onto_ontosyn_list:
        tfidf_ontoontosyn_rest.append(tf)
print('tfidf not in onto_ontosyn totally: %ld' % len(tfidf_ontoontosyn_rest))
# *******************onto，对比tfidf****************************
onto_tfidf_rest = []
for onto in onto_list:
    if onto not in tfidf_list:
        onto_tfidf_rest.append(onto)
print('onto not in tfidf totally: %ld' % len(onto_tfidf_rest))
# *******************onto，对比ontosyn****************************
onto_ontosyn_rest = []
for onto in onto_list:
    if onto not in ontosyn_list:
        onto_ontosyn_rest.append(onto)
print('onto not in ontosyn totally: %ld' % len(onto_ontosyn_rest))
# ************onto, 对比 合并tfidf + ontosyn**********************
tfidf_ontosyn_list = tfidf_list[:]
for ontosyn in ontosyn_list:
    if ontosyn not in tfidf_ontosyn_list:
        tfidf_ontosyn_list.append(ontosyn)
onto_tfidfontosyn_rest = []
for onto in onto_list:
    if onto not in tfidf_ontosyn_list:
        onto_tfidfontosyn_rest.append(onto)
print('onto not in tfidf_ontosyn totally: %ld' % len(onto_tfidfontosyn_rest))
# *******************ontosyn，对比tfidf****************************
ontosyn_tfidf_rest = []
for ontosyn in ontosyn_list:
    if ontosyn not in tfidf_list:
        ontosyn_tfidf_rest.append(ontosyn)
print('ontosyn not in tfidf totally: %ld' % len(ontosyn_tfidf_rest))
# *******************ontosyn，对比onto****************************
ontosyn_onto_rest = []
for ontosyn in ontosyn_list:
    if ontosyn not in onto_list:
        ontosyn_onto_rest.append(ontosyn)
print('ontosyn not in onto totally: %ld' % len(ontosyn_onto_rest))
# ************ontosyn, 对比 合并tfidf + onto**********************
tfidf_onto_list = tfidf_list[:]
for onto in onto_list:
    if onto not in tfidf_onto_list:
        tfidf_onto_list.append(onto)
ontosyn_tfidfonto_rest = []
for ontosyn in ontosyn_list:
    if ontosyn not in tfidf_onto_list:
        ontosyn_tfidfonto_rest.append(ontosyn)
print('ontosyn not in tfidf_onto totally: %ld' % len(ontosyn_tfidfonto_rest))
