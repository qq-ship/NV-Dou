from op_model.rhcp_model.testCode import pokerUtil as pokerUtils
from op_model.rhcp_model.testCode import dataToType as dataToType
from collections import Counter
import copy
from operator import itemgetter

MinCardsValue= -25
class COMB_TYPE:
    PASS, SOLO, PAIR, TRIO, TRIO_ONE, TRIO_TWO, SEQUENCE, SEQUENCE_TWO, SEQUENCE_THREE, BOMB, KING= range(
        11)
"""
获得该手牌（一次出牌）的价值
"""
def getOnePokerValue(onePoker):
    result=0
    onePokerType=dataToType.typeDic[onePoker][0]['type']
    if onePokerType==COMB_TYPE.KING:
        result=20
    else:
        maxCard=dataToType.typeDic[onePoker][0]['main']

        if type(maxCard)!=int:
            maxCard=pokerUtils.mapping(maxCard)
        if onePokerType==COMB_TYPE.PASS:
            result=0
        elif onePokerType==COMB_TYPE.SOLO:
            result=maxCard-10
        elif onePokerType==COMB_TYPE.PAIR:
            result=maxCard-10
        elif onePokerType==COMB_TYPE.TRIO:
            result=maxCard-10
        elif onePokerType==COMB_TYPE.SEQUENCE:
            result=maxCard-10+1
        elif onePokerType==COMB_TYPE.SEQUENCE_TWO:
            result = maxCard - 10 + 1
        elif onePokerType==COMB_TYPE.SEQUENCE_THREE:
            result=(maxCard-3+1)/2
        elif onePokerType==COMB_TYPE.TRIO_ONE:
            result = maxCard - 10
        elif onePokerType==COMB_TYPE.TRIO_TWO:
            result = maxCard - 10
        elif onePokerType==COMB_TYPE.BOMB:
            result = maxCard - 3+7
        else:
            print('get value error')
    return result

"""
获取到手上扑克的价值,如果是未出牌前，isOriginal=1（表示本轮过，因此比其他出牌多一轮） 其他情况为0
"""
def getPokerValue(poker,isOriginal):
    max=-1000
    result=chaifen(poker,100)
    for pokers in result:
        temp=0
        for p in pokers:
            temp=temp+getOnePokerValue(p)

        if max<temp-7*(len(pokers)+1):
            max=temp-7*(len(pokers)+1)
    return max

def get_all_legal_hands(pokerStr,showHands):
    result=[]
    pokerCounter=Counter(pokerStr)
    pokerCounterKeys=pokerCounter.keys()
    for hands in showHands:
        isExist=True
        handCounter=Counter(hands)
        for key,value in handCounter.items():
            if key not in pokerCounterKeys or value>pokerCounter[key]:
                isExist=False
                break
        if isExist:
            result.append(hands)
    return result
#根据当前扑克以及所有可能（合法）情况，选出当前扑克的合法情况
def get_all_split(allRulePoker,pokerStr,maxType):
    result=[]
    level=0
    recruimet(allRulePoker,pokerStr,0,[],result,maxType,level)
    result=sorted(result,key=lambda x:(len(x),pokerUtils.mapping(x[0])))
    #删除重复的结果
    delRepeat=[]
    for r in result:
        if r not in delRepeat:
            delRepeat.append(r)
    return delRepeat
def recruimet(allRulePoker,pokerString,startIndex,possibleLst,allPossibleLst,maxType,level):
    length = len(allRulePoker)
    if startIndex>=length:
        return
    if len(pokerString)==0:
        allPossibleLst.append(possibleLst)
        return
    for i in range(startIndex,length):
        currentPokerCount = Counter(pokerString)
        rulePokerCount = Counter(allRulePoker[i])
        isRight = True
        for key, value in rulePokerCount.items():
            if currentPokerCount[key] < value:
                isRight = False
                break
        if isRight == True:
            tmpPoker=pokerUtils.delPoker(pokerString,allRulePoker[i])
            tmpPossibleLst=copy.deepcopy(possibleLst)
            tmpPossibleLst.append(allRulePoker[i])
            recruimet(allRulePoker,tmpPoker,i,tmpPossibleLst,allPossibleLst,maxType,level+1)
        # #20181216 为避免不必要的拆分而增加
        # if len(allPossibleLst)>maxType:
        #     return


def chaifen(pokerStr,maxType):
    #获得所有的扑克组合
    allHands=dataToType.typeDic
    showHands=dataToType.allPokers
    #实现排序
    showHands=filter(lambda item:len(item)<=20,showHands)
    showHands = sorted(showHands,key = lambda i:len(i),reverse=True)
    #print(showHands)
    #根据当前扑克及所有的扑克组合，返回符合规则的扑克组合
    allRulePoker=get_all_legal_hands(pokerStr,showHands)
    allPossible=get_all_split(allRulePoker,pokerStr,maxType)
    return allPossible


def beidongPut(upCard,handCard):
    allPossiblePut=pokerUtils.getCanBeatPokers(upCard,handCard)
    #直接打光手牌
    if handCard in allPossiblePut:
        return handCard
    if 'L' in handCard and 'B' in handCard:
        tempHandCard = copy.deepcopy(handCard)
        tempHandCard = pokerUtils.delPoker(tempHandCard, 'L')
        tempHandCard = pokerUtils.delPoker(tempHandCard, 'B')
        if tempHandCard in dataToType.allPokers:
            return 'LB'
    sameType=[]
    bombType=[]
    #同类型牌压制
    upCardType=dataToType.typeDic[upCard][0]['type']
    for i in allPossiblePut:
        if dataToType.typeDic[i][0]['type'] ==upCardType:
            sameType.append(i)
        if dataToType.typeDic[i][0]['type'] in [9,10]:#9表示炸弹，10表示王炸
            bombType.append(i)
    #如果不出牌

    noPutvalue = getPokerValue(handCard, 1)
    ##
    dic={}
    maxKey=''
    sameType=sorted(sameType,key=lambda x:x[0])
    for i in sameType:
        tempHandCard=copy.deepcopy(handCard)
        tempHandCard=pokerUtils.delPoker(tempHandCard,i)
        value=getPokerValue(tempHandCard,0)
        if maxKey=='' or dic[maxKey]<value:
            maxKey=i
        if i not in dic.keys():
            dic[i]=value
        else:
            if dic[i]<value:
                dic[i] = value
    if maxKey in dic.keys() and dic[maxKey]>(noPutvalue-5):
        return maxKey


    #炸弹王炸压制
    dicBomb = {}
    maxKey = ''
    bombType = sorted(bombType, key=lambda x: x[0])
    for i in bombType:
        tempHandCard = copy.deepcopy(handCard)
        tempHandCard = pokerUtils.delPoker(tempHandCard, i)
        value = getPokerValue(tempHandCard, 0)
        if maxKey == '' or dicBomb[maxKey] < value:
            maxKey = i
        if i not in dicBomb.keys():
            dicBomb[i] = value
        else:
            if dicBomb[i] < value:
                dicBomb[i] = value
    if maxKey in dicBomb.keys() and dicBomb[maxKey] > (noPutvalue-5):
        return maxKey
    #不出
    return '0'

def custom_key(word):
    my_alphabet = ['3','4','5','6','7','8','9','T','J','Q','K','A','2','L','B']
    numbers = []
    for letter in word:
      numbers.append(my_alphabet.index(letter))
    return numbers

def zhudongPut(handCard):
    allCards=dataToType.allPokers

    if handCard in allCards:
        return handCard
    if 'L' in handCard and 'B' in handCard:
        tempHandCard = copy.deepcopy(handCard)
        tempHandCard = pokerUtils.delPoker(tempHandCard, 'L')
        tempHandCard = pokerUtils.delPoker(tempHandCard, 'B')
        if tempHandCard in dataToType.allPokers:
            return 'LB'
    possiblePutCard=[]
    handCardCounter=Counter(handCard)
    handCardCounterKeys=handCardCounter.keys()
    for card in allCards:

        isExist=True
        tempCounter=Counter(card)
        for key,value in tempCounter.items():
            if key not in handCardCounterKeys or value >handCardCounter[key]:
                isExist=False
                break
        if isExist:
            possiblePutCard.append(card)
    TRIO_ONE=[]
    TRIO_TWO=[]
    for putcard in possiblePutCard:
        if dataToType.typeDic[putcard][0]['type']==COMB_TYPE.TRIO_ONE:
            TRIO_ONE.append(putcard)
        elif dataToType.typeDic[putcard][0]['type']==COMB_TYPE.TRIO_TWO:
            TRIO_TWO.append(putcard)
    if len(TRIO_ONE)!=0:
        dictTrioOne={}
        maxKey=''
        for tr_one in TRIO_ONE:
            tempCard=copy.deepcopy(handCard)
            #print("tempCard --:",tempCard)
            tempCard=pokerUtils.delPoker(tempCard,tr_one)
            #print("tempCard --:", tempCard)
            value=getPokerValue(tempCard,0)
            #print("value --:", value)
            if maxKey=='':
                maxKey=tr_one
                dictTrioOne[tr_one]=value
            elif tr_one in dictTrioOne.keys():
                if dictTrioOne[tr_one]<value:
                    dictTrioOne[tr_one]=value
            else:
                dictTrioOne[tr_one] = value
                if dictTrioOne[maxKey]<value:
                    maxKey=tr_one
        return maxKey

    if len(TRIO_TWO)!=0:
        dictTrioTwo={}
        maxKey=''
        for tr_two in TRIO_TWO:
            tempCard=copy.deepcopy(handCard)
            tempCard=pokerUtils.delPoker(tempCard,tr_two)
            value=getPokerValue(tempCard,0)
            if maxKey=='':
                maxKey=tr_two
                dictTrioTwo[tr_two]=value
            elif tr_two in dictTrioTwo.keys():
                if dictTrioTwo[tr_two]<value:
                    dictTrioTwo[tr_two]=value
            else:
                dictTrioTwo[tr_two] = value
                if dictTrioTwo[maxKey]<value:
                    maxKey=tr_two
        return maxKey

    #出单牌
    single=[]
    for i in possiblePutCard:
        if len(i)==1 and handCard.count(i)!=4:
            single.append(i)
    single=sorted(single,key=custom_key)
    if len(single)!=0:
        return single[0]

    # 出对牌
    pair = []
    for i in possiblePutCard:
        if len(i) == 2 and handCard.count(i) != 4:
            pair.append(i)
    pair = sorted(pair, key=custom_key)
    if len(pair) != 0:
        return pair[0]

    # 出单顺
    sequence = []
    for i in possiblePutCard:
        if dataToType.typeDic[i]['type']==COMB_TYPE.SEQUENCE:
            sequence.append(i)
    sequence = sorted(sequence, key=custom_key)
    if len(sequence) != 0:
        return sequence[0]
    # 出三牌
    TRIO = []
    for i in possiblePutCard:
        if dataToType.typeDic[i]['type'] == COMB_TYPE.TRIO:
            TRIO.append(i)
    TRIO = sorted(TRIO, key=custom_key)
    if len(TRIO) != 0:
        return TRIO[0]
    # 出双顺
    sequence_two = []
    for i in possiblePutCard:
        if dataToType.typeDic[i]['type'] == COMB_TYPE.SEQUENCE_TWO:
            sequence_two.append(i)
    sequence_two = sorted(sequence_two, key=custom_key)
    if len(sequence_two) != 0:
        return sequence_two[0]
    # 出三顺
    sequence_TH = []
    for i in possiblePutCard:
        if dataToType.typeDic[i]['type'] == COMB_TYPE.SEQUENCE_THREE:
            sequence_TH.append(i)
        sequence_TH = sorted(sequence_TH, key=custom_key)
    if len(sequence_TH) != 0:
        return sequence_TH[0]

    # 出三顺
    bomb = []
    for i in possiblePutCard:
        if dataToType.typeDic[i]['type'] == COMB_TYPE.BOMB:
            bomb.append(i)
        bomb = sorted(bomb, key=custom_key)
    if len(bomb) != 0:
        return bomb[0]
    return possiblePutCard[0]

def getPutPoke(up2Poker,upPoker,handCard):
    realPoker='0'
    result='0'
    if upPoker=='0':
        realPoker=up2Poker
    else:
        realPoker=upPoker
    if realPoker=='0':
        result=zhudongPut(handCard)
    else:
        result=beidongPut(realPoker,handCard)
    return  result