import json
left_dict_path = 'F:\py programmes\programmes\OpenKE\left.json'
right_dict_path = 'F:\py programmes\programmes\OpenKE\\right.json'

file_path = 'F:\py programmes\programmes\OpenKE\ontology_analyse.txt'
left_dict = {}
right_dict = {}
with open(file_path, 'r') as f:
    for line in f:
        key = line[:line.index(':')]
        left = line[line.index(':')+1:line.index(' ')]
        right = line[line.rindex(' ')+1:line.index('\t')]
        left_dict[key] = left
        right_dict[key] = right
sort_left_dict = dict(sorted(left_dict.items(), key=lambda item:item[1], reverse=True))
sort_right_dict = dict(sorted(right_dict.items(), key=lambda item:item[1], reverse=True))
with open(left_dict_path, 'w') as f1:
    json.dump(sort_left_dict, f1)
print(sort_left_dict)
with open(right_dict_path, 'w') as f2:
    json.dump(sort_right_dict, f2)
print(sort_right_dict)