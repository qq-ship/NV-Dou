import json

if __name__ == "__main__":
    with open('./jsondata/abstract_map.json', 'r') as f:
        abstract_map = json.load(f)

    with open('./jsondata/action_space.json', 'r') as f:
        action_space = json.load(f)

    with open('./jsondata/card_type.json', 'r') as f:
        card_type = json.load(f)

    with open('./jsondata/specific_map.json', 'r') as f:
        specific_map = json.load(f)

    with open('./jsondata/type_card.json', 'r') as f:
        type_card = json.load(f)

    a_k_s = action_space.keys()
    all_put_cards = []
    i = 0
    zong = 0
    for m in a_k_s:
        print("第",i,"种类型个数:",len(abstract_map[m]))
        for cards in abstract_map[m]:
            all_put_cards.append(cards)


    """
    file = open('data.txt', 'w')
    file.write(str(all_put_cards));
    file.close()
    """


