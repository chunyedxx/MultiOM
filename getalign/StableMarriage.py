import json
import copy


def stable_marriage(dict):
    men_rankings = copy.deepcopy(dict['ma2nci'])
    women_rankings = copy.deepcopy(dict['nci2ma'])
    free_men = list(men_rankings.keys())
    free_women = list(women_rankings.keys())
    engaged = {}
    if len(men_rankings) <= len(women_rankings):
        while len(free_men) is not 0:
            m = free_men.pop(0)
            w = (men_rankings[m]).pop(0)
            if w in free_women:
                engaged[m] = w
                free_women.remove(w)
            else:
                for k, v in engaged.items():
                    if w in v:
                        mm = k
                if women_rankings[w].index(m) < women_rankings[w].index(mm):
                    engaged[m] = w
                    del engaged[mm]
                    free_men.append(mm)
                else: free_men.append(m)
    else:
        while len(free_women) is not 0:
            w = free_women.pop(0)
            m = (women_rankings[w]).pop(0)
            if m in free_men:
                engaged[w] = m
                free_men.remove(m)
            else:
                for k, v in engaged.items():
                    if m in v:
                        ww = k
                if men_rankings[m].index(w) < men_rankings[m].index(ww):
                    engaged[w] = m
                    del engaged[ww]
                    free_women.append(ww)
                else: free_women.append(w)
    return engaged
