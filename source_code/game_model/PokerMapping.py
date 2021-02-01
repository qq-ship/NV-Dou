import numpy as np
from rlcard.games.doudizhu.utils import SPECIFIC_MAP, ACTION_SPACE,ABSTRACT_MAP
ACTION_ID_TO_STR =  {v: k for k, v in ACTION_SPACE.items()}

NumtoCard = {0:'3',1:'4',2:'5',3:'6',4:'7',5:'8',6:'9',7:'T',8:'J',9:'Q',10:'K',11:'A',12:'2',13:'B',14:'R'}
cardsIndex = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,'2': 12,'B': 13, 'R': 14}
cardsIndexRH = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,'2': 12, '*': 13, '$': 14}
CardsLeixing = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 2, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 4, 55: 4, 56: 4, 57: 4, 58: 4, 59: 4, 60: 4, 61: 4, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 67: 5, 68: 5, 69: 5, 70: 5, 71: 5, 72: 5, 73: 5, 74: 5, 75: 5, 76: 5, 77: 5, 78: 5, 79: 5, 80: 5, 81: 5, 82: 5, 83: 5, 84: 5, 85: 5, 86: 5, 87: 5, 88: 5, 89: 5, 90: 5, 91: 5, 92: 5, 93: 5, 94: 5, 95: 5, 96: 5, 97: 5, 98: 5, 99: 5, 100: 5, 101: 5, 102: 5, 103: 6, 104: 6, 105: 6, 106: 6, 107: 6, 108: 6, 109: 6, 110: 6, 111: 6, 112: 6, 113: 6, 114: 6, 115: 6, 116: 6, 117: 6, 118: 6, 119: 6, 120: 6, 121: 6, 122: 6, 123: 6, 124: 6, 125: 6, 126: 6, 127: 6, 128: 6, 129: 6, 130: 6, 131: 6, 132: 6, 133: 6, 134: 6, 135: 6, 136: 6, 137: 6, 138: 6, 139: 6, 140: 6, 141: 6, 142: 6, 143: 6, 144: 6, 145: 6, 146: 6, 147: 6, 148: 6, 149: 6, 150: 6, 151: 6, 152: 6, 153: 6, 154: 6, 155: 7, 156: 7, 157: 7, 158: 7, 159: 7, 160: 7, 161: 7, 162: 7, 163: 7, 164: 7, 165: 7, 166: 7, 167: 7, 168: 7, 169: 7, 170: 7, 171: 7, 172: 7, 173: 7, 174: 7, 175: 7, 176: 7, 177: 7, 178: 7, 179: 7, 180: 7, 181: 7, 182: 7, 183: 7, 184: 7, 185: 7, 186: 7, 187: 7, 188: 7, 189: 7, 190: 7, 191: 7, 192: 7, 193: 7, 194: 7, 195: 7, 196: 7, 197: 7, 198: 7, 199: 7, 200: 8, 201: 8, 202: 8, 203: 8, 204: 8, 205: 8, 206: 8, 207: 8, 208: 8, 209: 8, 210: 8, 211: 8, 212: 8, 213: 8, 214: 8, 215: 8, 216: 8, 217: 8, 218: 8, 219: 8, 220: 8, 221: 8, 222: 8, 223: 8, 224: 8, 225: 8, 226: 8, 227: 8, 228: 8, 229: 8, 230: 8, 231: 8, 232: 8, 233: 8, 234: 8, 235: 8, 236: 8, 237: 8, 238: 9, 239: 9, 240: 9, 241: 9, 242: 9, 243: 9, 244: 9, 245: 9, 246: 9, 247: 9, 248: 9, 249: 9, 250: 9, 251: 9, 252: 9, 253: 9, 254: 9, 255: 9, 256: 9, 257: 9, 258: 9, 259: 9, 260: 9, 261: 9, 262: 9, 263: 9, 264: 9, 265: 9, 266: 9, 267: 9, 268: 10, 269: 10, 270: 10, 271: 10, 272: 10, 273: 10, 274: 10, 275: 10, 276: 10, 277: 10, 278: 10, 279: 10, 280: 10, 281: 11, 282: 11, 283: 11, 284: 11, 285: 11, 286: 11, 287: 11, 288: 11, 289: 11, 290: 11, 291: 11, 292: 11, 293: 11, 294: 12, 295: 12, 296: 12, 297: 12, 298: 12, 299: 12, 300: 12, 301: 12, 302: 12, 303: 12, 304: 12, 305: 12, 306: 12, 307: 13, 308: 14}
leixingstr = {0:'单牌',1:'对牌',2:'对三',3:'三带一',4:'三带一对',5:'单连',6:'连对',7:'连三',8:'飞机带单',9:'飞机带对',10:'四带两单',11:'四带两对',12:'炸弹',13:'火箭',14:'PASS'}

def duotodan(tt):
    rs = []

    for ts in tt:
        tmp = np.delete(ts,0,axis= 0)
        for i in range(len(tmp)):
            tmp[i] = tmp[i] * (i+1)
        tmp01 = tmp.sum(axis=0)
        rs.append(tmp01)
    return np.array(rs)


def numpytostr(state):
    handcard = np.array(state["obs"][0])
    other_two_card = np.array(state["obs"][1])
    one_handcard = np.array(state["obs"][2])
    two_handcard = np.array(state["obs"][3])
    three_handcard = np.array(state["obs"][4])
    palyed_card = np.array(state["obs"][5])


    legal_actions = np.array(state["legal_actions"])
    cardstr =""
    other_two_action = ""
    one_last_action = ""
    two_last_action = ""
    three_last_action=""
    palyed_action = ""

    for i in range(15):
        cardnum = handcard[:,i].tolist().index(1)
        othernum = other_two_card[:,i].tolist().index(1)
        one_cardnum = one_handcard[:, i].tolist().index(1)
        two_cardnum = two_handcard[:, i].tolist().index(1)
        three_cardnum = three_handcard[:, i].tolist().index(1)
        palyed_num = palyed_card[:, i].tolist().index(1)

        if cardnum != 0:
            for mm in range(cardnum):
                cardstr = cardstr + NumtoCard[i]

        if othernum != 0:
            for mm in range(othernum):
                other_two_action = other_two_action + NumtoCard[i]

        if one_cardnum != 0:
            for mm in range(one_cardnum):
                one_last_action = one_last_action + NumtoCard[i]

        if two_cardnum != 0:
            for mm in range(two_cardnum):
                two_last_action = two_last_action + NumtoCard[i]

        if three_cardnum != 0:
            for mm in range(three_cardnum):
                three_last_action = three_last_action + NumtoCard[i]

        if palyed_num != 0:
            for mm in range(palyed_num):
                palyed_action = palyed_action + NumtoCard[i]


    return cardstr,other_two_action,one_last_action,two_last_action,three_last_action,palyed_action

def strtonumpy(carddic):

    card_numpy = np.zeros((5,15))
    card_numpy[0,:] = 1

    if len(carddic) == 0:
        return card_numpy
    else:
        keys = list(carddic.keys())
        for k in keys:
            kid = cardsIndex[k]
            knum = carddic[k]
            card_numpy[0][kid] = 0
            card_numpy[knum][kid] = 1
        return card_numpy

def rltorh(card):
    curreny_hand = []
    for i in range(len(card)):
        if card[i] is 'T':
            curreny_hand.append('10')
        elif card[i] is 'B':
            curreny_hand.append('*')
        elif card[i] is 'R':
            curreny_hand.append('$')
        else:
            curreny_hand.append(card[i])
    return curreny_hand
def rhtorl(putcard):
    if len(putcard) == 0:
        action = 'pass'
    else:
        action = ''
        cardindex = []
        for i in range(len(putcard)):
            cardindex.append(cardsIndexRH[putcard[i]])
        cardindex.sort()

        for index in cardindex :
            action = action + NumtoCard[index]

    return action
def rhnumtorl(rhnums):
    type = rhnums[0]
    max = rhnums[1]
    num = rhnums[2]

    if type == -1:
        return -1

    elif type == 0:
        return 308

    elif type == 1:
        return max-3

    elif type == 2:
        return max + 12
    elif type == 3:
        return max + 25
    elif type == 4:
        return max + 291
    elif type == 5:
        return max + 38
    elif type == 6:
        return max + 51
    elif type == 7:

        qishi = max - num + 1
        mystr = ""
        for i in range(qishi,max+1):
            if i == 10:
                mystr = mystr + 'T'
            elif i == 11:
                mystr = mystr + 'J'
            elif i == 12:
                mystr = mystr + 'Q'
            elif i == 13:
                mystr = mystr + 'K'
            elif i == 14:
                mystr = mystr + 'A'
            elif i == 15:
                mystr = mystr + '2'
            elif i == 16:
                mystr = mystr + 'B'
            elif i == 17:
                mystr = mystr + 'R'
            else:
                mystr = mystr + str(i)

        ac = ACTION_SPACE[mystr]
        return ac

    elif type == 8:
        num = int(num / 2)
        qishi = max - num + 1
        mystr = ""
        for i in range(qishi,max+1):
            if i == 10:
                mystr = mystr + 'T'
                mystr = mystr + 'T'
            elif i == 11:
                mystr = mystr + 'J'
                mystr = mystr + 'J'
            elif i == 12:
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
            elif i == 13:
                mystr = mystr + 'K'
                mystr = mystr + 'K'
            elif i == 14:
                mystr = mystr + 'A'
                mystr = mystr + 'A'
            elif i == 15:
                mystr = mystr + '2'
                mystr = mystr + '2'
            elif i == 16:
                mystr = mystr + 'B'
                mystr = mystr + 'B'
            elif i == 17:
                mystr = mystr + 'R'
                mystr = mystr + 'R'
            else:
                mystr = mystr + str(i)
                mystr = mystr + str(i)

        ac = ACTION_SPACE[mystr]
        return ac

    elif type == 9:
        num = int(num / 3)
        qishi = max - num + 1
        mystr = ""
        for i in range(qishi, max + 1):
            if i == 10:
                mystr = mystr + 'T'
                mystr = mystr + 'T'
                mystr = mystr + 'T'
            elif i == 11:
                mystr = mystr + 'J'
                mystr = mystr + 'J'
                mystr = mystr + 'J'
            elif i == 12:
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
            elif i == 13:
                mystr = mystr + 'K'
                mystr = mystr + 'K'
                mystr = mystr + 'K'
            elif i == 14:
                mystr = mystr + 'A'
                mystr = mystr + 'A'
                mystr = mystr + 'A'
            elif i == 15:
                mystr = mystr + '2'
                mystr = mystr + '2'
                mystr = mystr + '2'

            elif i == 16:
                mystr = mystr + 'B'
                mystr = mystr + 'B'
                mystr = mystr + 'B'
            elif i == 17:
                mystr = mystr + 'R'
                mystr = mystr + 'R'
                mystr = mystr + 'R'
            else:
                mystr = mystr + str(i)
                mystr = mystr + str(i)
                mystr = mystr + str(i)
        ac = ACTION_SPACE[mystr]
        return ac
    elif type == 10:
        num = int(num / 4)
        qishi = max - num + 1
        mystr = ""
        for i in range(qishi, max + 1):
            if i == 10:
                mystr = mystr + 'T'
                mystr = mystr + 'T'
                mystr = mystr + 'T'
            elif i == 11:
                mystr = mystr + 'J'
                mystr = mystr + 'J'
                mystr = mystr + 'J'
            elif i == 12:
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
            elif i == 13:
                mystr = mystr + 'K'
                mystr = mystr + 'K'
                mystr = mystr + 'K'
            elif i == 14:
                mystr = mystr + 'A'
                mystr = mystr + 'A'
                mystr = mystr + 'A'
            elif i == 15:
                mystr = mystr + '2'
                mystr = mystr + '2'
                mystr = mystr + '2'

            elif i == 16:
                mystr = mystr + 'B'
                mystr = mystr + 'B'
                mystr = mystr + 'B'
            elif i == 17:
                mystr = mystr + 'R'
                mystr = mystr + 'R'
                mystr = mystr + 'R'
            else:
                mystr = mystr + str(i)
                mystr = mystr + str(i)
                mystr = mystr + str(i)
        ac = ACTION_SPACE[mystr] + 45
        return ac
    elif type == 11:
        num = int(num / 5)
        qishi = max - num + 1
        mystr = ""
        for i in range(qishi, max + 1):
            if i == 10:
                mystr = mystr + 'T'
                mystr = mystr + 'T'
                mystr = mystr + 'T'
            elif i == 11:
                mystr = mystr + 'J'
                mystr = mystr + 'J'
                mystr = mystr + 'J'
            elif i == 12:
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
                mystr = mystr + 'Q'
            elif i == 13:
                mystr = mystr + 'K'
                mystr = mystr + 'K'
                mystr = mystr + 'K'
            elif i == 14:
                mystr = mystr + 'A'
                mystr = mystr + 'A'
                mystr = mystr + 'A'
            elif i == 15:
                mystr = mystr + '2'
                mystr = mystr + '2'
                mystr = mystr + '2'

            elif i == 16:
                mystr = mystr + 'B'
                mystr = mystr + 'B'
                mystr = mystr + 'B'
            elif i == 17:
                mystr = mystr + 'R'
                mystr = mystr + 'R'
                mystr = mystr + 'R'
            else:
                mystr = mystr + str(i)
                mystr = mystr + str(i)
                mystr = mystr + str(i)

        ac = ACTION_SPACE[mystr] + 83
        return ac
    elif type == 12:
        return 307
    elif type == 13:
        return max + 265
    elif type == 14:
        return max + 278
    elif type == 0:

        return -1


def cardcouple(type,maxcard,num):
    couples = []
    for i in range(len(type)):
        rcard = rhnumtorl([type[i],maxcard[i],num[i]])
        couples.append(rcard)
    return couples


def actiontostr(actions):
    print("---")

def strtoaction(actions):
    print("####")

dic  = {"3":4,"A":3,"2":2,"B":1}
if __name__ == "__main__":
    strtonumpy(dic)








