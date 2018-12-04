import pandas as pd
import numpy as np
from scipy.linalg import norm
import path_file
import gc
import w2v_util
import json

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
dict_txt = path_file.dict_new_txt
title_prediction_distance_txt = path_file.title_prediction_distance_new_txt

sentence_to_vec_dict = {}

train_data = pd.read_table(train_txt,
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8',quoting=3).astype(str)
val_data = pd.read_table(val_txt,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8',quoting=3).astype(str)
test_data = pd.read_table(test_txt,
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8',quoting=3).astype(str)
#去噪，只有一条数据
train_data = train_data[train_data['label'] != '音乐' ]
#统一赋值test_data的label
test_data['label'] = -1

#连接在一起便于统一处理
data = pd.concat([train_data,val_data,test_data],ignore_index=True)
data['label'] = data['label'].apply(lambda x: int(x))

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

del train_data
del val_data
del test_data
gc.collect()

##########构造的特征###########
title_prediction_distance = pd.DataFrame()

def sentence_vec(str):
    if str in sentence_to_vec_dict:
        return np.array(sentence_to_vec_dict[str])
    vec = w2v_util.sentence_vec(str)
    sentence_to_vec_dict[str] = vec.tolist()
    return vec

def distance(v1,v2):
    #余弦距离
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")

data["query_prediction"] = data["query_prediction"].apply(loads)

title_tag_distances = []

title_prediction_distance_means = []
title_prediction_distance_stds = []
title_prediction_distance_mins = []
title_prediction_distance_maxs = []

title_prediction_distance_w_means = []
title_prediction_distance_w_mean_news = []

title_max_prediction_distances = []

for i,row in data.iterrows():
    if i%1000 == 0:
        print(i)

    # s：query_prediction字符串
    s = row['query_prediction']

    title = str(row['title'])
    title_vec = sentence_vec(title)

    tag = str(row['tag'])
    tag_vec = sentence_vec(tag)

    title_tag_distance = distance(title_vec, tag_vec)

    # max_pre_title和max_pre_title_score记录预测词组中最有可能的词组以及它的概率
    max_pre_title = ''
    max_pre_title_score = 0

    all_pre = []
    all_distance = []
    pre_dict = {}

    for query_item, query_ratio in s.items():
        all_pre.append(float(query_ratio))
        all_distance.append(distance(sentence_vec(str(query_item)), title_vec))
        pre_dict[str(query_item)] = float(query_ratio)

        if float(query_ratio) > max_pre_title_score:
            max_pre_title_score = float(query_ratio)
            max_pre_title = str(query_item)

    # 对概率统计
    is_null = len(all_pre)

    title_prediction_distance_mean = 1
    title_prediction_distance_std = 1
    title_prediction_distance_min = 1
    title_prediction_distance_max = 1

    title_prediction_distance_w_mean = 1
    title_prediction_distance_w_mean_new = 1
    title_max_prediction_distance = 1

    if is_null > 0:
        if np.sum(all_pre) == 0:
            print(s)
            print(all_pre)

        title_prediction_distance_mean = np.mean(all_distance)
        title_prediction_distance_std = np.std(all_distance)
        title_prediction_distance_min = np.min(all_distance)
        title_prediction_distance_max = np.max(all_distance)

        title_prediction_distance_w_mean = np.sum(np.array(all_distance) * np.array(all_pre)) / np.sum(all_pre)
        title_prediction_distance_w_mean_new = np.sum(np.array(all_distance) * np.array(all_pre))
        title_max_prediction_distance = distance(sentence_vec(max_pre_title), title_vec)

    title_tag_distances.append(title_tag_distance)

    title_prediction_distance_means.append(title_prediction_distance_mean)
    title_prediction_distance_stds.append(title_prediction_distance_std)
    title_prediction_distance_mins.append(title_prediction_distance_min)
    title_prediction_distance_maxs.append(title_prediction_distance_max)

    title_prediction_distance_w_means.append(title_prediction_distance_w_mean)
    title_prediction_distance_w_mean_news.append(title_prediction_distance_w_mean_new)

    title_max_prediction_distances.append(title_max_prediction_distance)

title_prediction_distance['title_tag_distance'] = title_tag_distances

title_prediction_distance['title_prediction_distance_mean'] = title_prediction_distance_means
title_prediction_distance['title_prediction_distance_std'] = title_prediction_distance_stds
title_prediction_distance['title_prediction_distance_min'] = title_prediction_distance_mins
title_prediction_distance['title_prediction_distance_max'] = title_prediction_distance_maxs

title_prediction_distance['title_prediction_distance_w_mean'] = title_prediction_distance_w_means
title_prediction_distance['title_prediction_distance_w_mean_new'] = title_prediction_distance_w_mean_news

title_prediction_distance['title_max_prediction_distance'] = title_max_prediction_distances

print (title_prediction_distance.shape)

title_prediction_distance.to_csv(title_prediction_distance_txt, sep='\t', index=False)

# import json
#
# jsObj = json.dumps(sentence_to_vec_dict)
#
# fileObject = open(dict_txt, 'w')
# fileObject.write(jsObj)
# fileObject.close()


