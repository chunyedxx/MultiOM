import json


def align_eval(alignpath, referpath):
    corres = 0
    unknow = 0
    with open(alignpath, "r", encoding='utf-8') as f:
        alignments = []
        for line in f:
            left, right, _ = line.strip().split(',')
            if (left, right) not in alignments:
                alignments.append((left, right))
    with open(referpath, "r", encoding='utf-8') as f:
        referencemaps = list(f)
    refernum = 0
    for line in referencemaps:
        if '?' not in line:
            refernum += 1
    for map in alignments:
        if map[0] + ',' + map[1] + ',=\n' in referencemaps:
            corres += 1
        if map[1] + ',' + map[0] + ',?\n' in referencemaps:
            unknow += 1

    if 'MA2NCI' in alignpath:
        print('align num：%ld' % len(alignments))
        print("corres num：%ld" % corres)
        pre = corres / len(alignments)
        rec = corres / len(referencemaps)
        F = pre * rec * 2 / (pre + rec)
        print("pre :%f \t rec:%f \t F:%f" % (pre, rec, F))
    elif 'FMA2NCI' in alignpath:
        print('匹配总数：%ld' % len(alignments))
        print("正确个数：%ld" % corres)
        pre = corres / (len(alignments) - unknow)
        rec = corres / refernum
        F = pre * rec * 2 / (pre + rec)
        print("pre :%f \t rec:%f \t F:%f" % (pre, rec, F))


if __name__ == '__main__':
    alignpath = '.\\align_ontosyn_all.txt'
    referpath = '.\\reference_fma2nci.txt'
    align_eval(alignpath, referpath)
