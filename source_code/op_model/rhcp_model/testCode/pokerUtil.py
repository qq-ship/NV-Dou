from op_model.rhcp_model.testCode.dataToType import typeDic
from functools import cmp_to_key# 配置出牌规则
from collections import Counter

def mapping(poker):
    if len(poker)>1:
        poker=poker[0]
    if poker>'2' and poker<='9':
        return int(poker)
    elif poker=='T':
        return 10
    elif poker=='J':
        return 11
    elif poker=='Q':
        return 12
    elif poker=='K':
        return 13
    elif poker=='A':
        return 14
    elif poker=='2':
        return 15
    elif poker=='L':
        return 16
    elif poker=='B':
        return 17
    else:
        return 17

class COMB_TYPE:
    PASS, SOLO,PAIR,TRIO,TRIO_ONE,TRIO_TWO,SEQUENCE,SEQUENCE_TWO,SEQUENCE_THREE,BOMB,KING = range(
        11)

def cmp(x,y):
    return mapping(x)-mapping(y)

def sortString(string):
    data=sorted(str(string), key=cmp_to_key(cmp))
    string=""
    for s in data:
        string=string+s
    return string


#依据上家真实出牌（有可能是上上家的出牌），求出当前手牌中，可能压过真实上家的出牌
def getCanBeatPokers(realUpPoker,currentPokers):
    possibleBeatPokers=[]
    counterCurrentPokers = Counter(currentPokers)
    ccpKeys=counterCurrentPokers.keys()
    for key in typeDic.keys():
        counterKey = Counter(key)
        #是否满足组合形式（手牌是否能组合成）
        isCanZhuhe=True
        for k, v in counterKey.items():
            if not(k in ccpKeys and v <= counterCurrentPokers[k]):
                isCanZhuhe=False
                break
        if isCanZhuhe and can_comb2_beat_comb1(realUpPoker,key):
            possibleBeatPokers.append(key)

    return possibleBeatPokers


def delPoker(handPoker,putPoker):
    for i in putPoker:
        if i!='0':
            handPoker=handPoker.replace(i,'',1)
    return handPoker

def can_comb2_beat_comb1(comb1, comb2):
    comb1=sortString(comb1)
    comb2 = sortString(comb2)
    comb1=typeDic[comb1]
    comb2=typeDic[comb2]
    comb1=comb1[0]
    comb2 = comb2[0]
    if comb2['type'] == COMB_TYPE.PASS:
        return False

    if not comb1 or comb1['type'] == COMB_TYPE.PASS:
        return True

    if comb1['type'] == comb2['type']:
        # print(comb1)
        # print(comb2)
        cb1 = 0
        cb2 = 0
        if type(comb1['main']) != int:
            cb1 = mapping(comb1['main'])
        else:
            cb1 = comb1['main']
        if type(comb2['main']) != int:
            cb2 = mapping(comb2['main'])
        else:
            cb2 = comb2['main']
        if comb1['type'] == COMB_TYPE.SEQUENCE:
            if cb1 != cb2:
                return False
            else:
                return mapping(comb2['sub']) > mapping(comb1['sub'])
        else:
            if cb1 == cb2 and comb1['type']!=COMB_TYPE.PAIR and comb1[
                'type']!=COMB_TYPE.TRIO and comb1['type']!=COMB_TYPE.SOLO:
                return cb2 > cb1
            else:

                return cb2 > cb1
    elif comb2['type'] == COMB_TYPE.BOMB or comb2['type'] == COMB_TYPE.KING:
        return comb2['type'] > comb1['type']

    return False