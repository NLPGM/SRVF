import json

if __name__ == '__main__':
    with open('test.json', encoding="utf-8", mode='r') as f:
        data = json.load(f)
    groups = {}
    for item in data:
        relation = item['relation_type']
        if relation not in groups:
            groups[relation] = []
        groups[relation].append(item)

    num_detail = {}
    for key in groups.keys():
        num_detail[key] = len(groups[key])

    for key in num_detail.keys():
        print(key, '|', num_detail[key])

    print(len(num_detail.keys()))
    print(num_detail)
    print('len(data)/len(num_detail.keys())',len(data)/len(num_detail.keys()))

