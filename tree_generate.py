from treelib import Tree
train2id_path = 'D:\ontomap_3\\benchmarks\DXX\DXX_FMA\\train2id.txt'
train2id_list = []
father_list = []
son_list = []
with open(train2id_path, "r") as f:
    for line in f:
        son = (line[:line.index(' ')])
        if son not in son_list:
            son_list.append(son)
        father = (line[line.index(' ') + 1:line.rindex(' ')])
        if father not in father_list:
            father_list.append(father)
        train2id_list.append((son, father))
for father_ in father_list:
    if father_ not in son_list:
        print(father_)

# 根节点为:65315，61220，11160，10457
tree = Tree()
gen_ = "10457"
tree.create_node("Root" + ":" + gen_, gen_)


def cycle_search(i=0):
    j = i
    for parent in train2id_list:
        if tree.contains(parent[0]):
            if tree.parent(parent[0]).identifier == parent[1]:
                continue
        if tree.contains(parent[1]):
            tree.create_node("Children"+":" + parent[0], parent[0], parent=parent[1])
            print(i)
            print(parent)
            i += 1
    if j == i:
        return tree
    return cycle_search(i)


a = cycle_search()
print(a)
print("the tree depth is: %ld" % a.depth())

