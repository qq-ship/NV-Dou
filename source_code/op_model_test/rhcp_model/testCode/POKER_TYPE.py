# 定义牌型
"""
编号 牌型 描述和备注
0 弃权 无出牌
1 单牌 一张单牌（型如A）
2 对牌 点数相同的两张牌（型如AA）
3 三条 点数相同的三张牌（型如AAA）
4,5 三带一 三条带一张单牌或一对牌。（型如：AAA+B或AAA+BB）
6 单顺 五张或更多的连续单牌不包括2和王（型如：ABCDE或ABCDE...）
7 双顺 三对或更多的连续对牌不包括2和王（型如：AABBCC或AABBCC..）
8 三顺 二个或更多的连续三条不包括2和王（型如：AAABBB或AAABBBCCC...）
9,10 三顺带牌 三顺带同数量的单牌或同数量的对牌（型如：AAABBB+C+D或AAABBB+CC+DD或AAABBB... +...+Y+Z或AAABBB... +...+YY+ZZ）
11,12 四带二 四张同点数牌带2张单牌或2对牌（型如AAAA+B+C或AAAA+BB+CC）
13 炸弹 四张同点数牌（型如AAAA）
14 火箭 双王（大王和小王），最大的牌型
-1 非法牌型
以上牌型以外的牌张组合
"""
class POKER_TYPE:
    PASS, SINGLE, PAIR, TRIPLE, TRIPLE_ONE, TRIPLE_TWO, STRIGHT,STRIGHT_DOUBLE,STRIGHT_TRIPLE,STRIGHT_TRIPLE_ONES,STRIGHT_TRIPLE_PAIRS,\
    FOURTH_TWO_ONES, FOURTH_TWO_PAIRS,  BOMB, KING_PAIR= range(15)
