import pandas as pd
import numpy as np
import gc

train_txt = '../DataSets/oppo_data_ronud2_20181107/data_train.txt'
val_txt = '../DataSets/oppo_data_ronud2_20181107/data_vali.txt'
test_txt = '../DataSets/oppo_data_ronud2_20181107/data_test.txt'
train_data = pd.read_table(train_txt,
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8',quoting=3).astype(str)
val_data = pd.read_table(val_txt,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8',quoting=3).astype(str)
test_data = pd.read_table(test_txt,
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8',quoting=3).astype(str)

len_train = train_data.shape[0]
len_val = val_data.shape[0]
len_test = test_data.shape[0]
print("len_train",len_train)
print("len_val",len_val)
print("len_test",len_test)

#连接在一起便于统一处理
data = pd.concat([train_data,val_data,test_data],ignore_index=True)
len_data = data.shape[0]
del train_data
del val_data
del test_data
gc.collect()

import thulac
thu1 = thulac.thulac()  #默认模式

def sentence_cut(s):
    s = s.lower()
    cut = thu1.cut(s, text=False)
#     print(cut)
    items = []
    for item in cut:
        items.append(item[0])
    return items

import json
def loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")
data["query_prediction"] = data["query_prediction"].apply(loads)

cut_dict = dict()
ith_data = data[2000000:]

for j,row in ith_data.iterrows():
    if j % 1000 == 0:
        print(j)
    title = str(row['title'])
    prefix = str(row['prefix'])
    predict = row['query_prediction']
    if title not in cut_dict:
        cut_dict[title] = sentence_cut(title)
    if prefix not in cut_dict:
        cut_dict[prefix] = sentence_cut(prefix)
    for query_item, query_ratio in predict.items():
        if str(query_item) not in cut_dict:
            cut_dict[str(query_item)] = sentence_cut(str(query_item))
jsObj = json.dumps(cut_dict)
fileObject = open(str(2000000) + '_'+str(2250000)+'.json', 'w')
fileObject.write(jsObj)
fileObject.close()