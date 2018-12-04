import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import path_file
import gc
import json

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
title_prediction_jaccard_distance_txt = path_file.title_prediction_jaccard_distance_new_txt

train_data = pd.read_table(train_txt,
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8',quoting=3).astype(str)
val_data = pd.read_table(val_txt,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8',quoting=3).astype(str)
test_data = pd.read_table(test_txt,
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8',quoting=3).astype(str)
#去噪，只有一条数据
train_data = train_data[train_data['label'].isin(['0', '1'])]
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


def add_space(s):
    return ' '.join(list(s))

def jaccard_distance(s1,s2):
    def add_space(s):
        return ' '.join(list(s))
    #将字母大小写统一为小写
    s1 = s1.lower()
    s2 = s2.lower()
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德距离

    return 1.0 - 1.0 * numerator / denominator

def loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")

data["query_prediction"] = data["query_prediction"].apply(loads)

title_is_all_zhs = []

title_prediction_jaccard_distance_means = []
title_prediction_jaccard_distance_stds = []
title_prediction_jaccard_distance_mins = []
title_prediction_jaccard_distance_maxs = []
title_prediction_jaccard_distance_w_means = []
title_prediction_jaccard_distance_w_mean_news = []

title_max_prediction_jaccard_distances = []

data['title_digit_len'] = data['title'].apply(lambda x: sum(c.isdigit() for c in x))
data['title_alpha_len'] = data['title'].apply(lambda x: len(re.findall('[a-zA-Z]',x)))
# data['title'] = data['title'].apply(lambda x: add_space(x.lower()))

for i,row in data.iterrows():
    if i%1000 == 0:
        print(i)
    # s：query_prediction字符串
    s = row['query_prediction']
    title = str(row['title'])

    title_is_all_zh = 1
    if row['title_digit_len'] > 0 or row['title_alpha_len'] > 0:
        title_is_all_zh = 0

    # max_pre_title和max_pre_title_score记录预测词组中最有可能的词组以及它的概率
    max_pre_title = ''
    max_pre_title_score = 0

    all_pre = []
    all_jaccard_distance = []
    pre_dict = {}

    for query_item, query_ratio in s.items():
        all_pre.append(float(query_ratio))
        # query_item = add_space(str(query_item).lower())
        all_jaccard_distance.append(jaccard_distance(title, str(query_item)))
        pre_dict[str(query_item)] = float(query_ratio)

        if float(query_ratio) > max_pre_title_score:
            max_pre_title_score = float(query_ratio)
            max_pre_title = str(query_item)

    ##############################################
    ##pre feat
    ##############################################

    # 对概率统计
    is_null = len(all_pre)

    title_prediction_jaccard_distance_mean = 1
    title_prediction_jaccard_distance_std = 1
    title_prediction_jaccard_distance_min = 1
    title_prediction_jaccard_distance_max = 1

    title_prediction_jaccard_distance_w_mean = 1
    title_prediction_jaccard_distance_w_mean_new = 1
    title_max_prediction_jaccard_distance = 1
    if is_null > 0:
        if np.sum(all_pre) == 0:
            print(s)
            print(all_pre)

        # 如果prediction不为""或者"{}"
        title_prediction_jaccard_distance_mean = np.mean(all_jaccard_distance)
        title_prediction_jaccard_distance_std = np.std(all_jaccard_distance)
        title_prediction_jaccard_distance_min = np.min(all_jaccard_distance)
        title_prediction_jaccard_distance_max = np.max(all_jaccard_distance)

        title_prediction_jaccard_distance_w_mean = np.sum(np.array(all_jaccard_distance) * np.array(all_pre)) / np.sum(all_pre)
        title_prediction_jaccard_distance_w_mean_new = np.sum(np.array(all_jaccard_distance) * np.array(all_pre))
        title_max_prediction_jaccard_distance = jaccard_distance(max_pre_title, title)

    title_is_all_zhs.append(title_is_all_zh)

    title_prediction_jaccard_distance_means.append(title_prediction_jaccard_distance_mean)
    title_prediction_jaccard_distance_stds.append(title_prediction_jaccard_distance_std)
    title_prediction_jaccard_distance_mins.append(title_prediction_jaccard_distance_min)
    title_prediction_jaccard_distance_maxs.append(title_prediction_jaccard_distance_max)

    title_prediction_jaccard_distance_w_means.append(title_prediction_jaccard_distance_w_mean)
    title_prediction_jaccard_distance_w_mean_news.append(title_prediction_jaccard_distance_w_mean_new)

    title_max_prediction_jaccard_distances.append(title_max_prediction_jaccard_distance)

title_prediction_distance['title_is_all_zh'] = title_is_all_zhs

title_prediction_distance['title_prediction_jaccard_distance_mean'] = title_prediction_jaccard_distance_means
title_prediction_distance['title_prediction_jaccard_distance_std'] = title_prediction_jaccard_distance_stds
title_prediction_distance['title_prediction_jaccard_distance_min'] = title_prediction_jaccard_distance_mins
title_prediction_distance['title_prediction_jaccard_distance_max'] = title_prediction_jaccard_distance_maxs

title_prediction_distance['title_prediction_jaccard_distance_w_mean'] = title_prediction_jaccard_distance_w_means
title_prediction_distance['title_prediction_jaccard_distance_w_mean_new'] = title_prediction_jaccard_distance_w_mean_news
title_prediction_distance['title_max_prediction_jaccard_distance'] = title_max_prediction_jaccard_distances

print (title_prediction_distance.shape)

title_prediction_distance.to_csv(title_prediction_jaccard_distance_txt, sep='\t', index=False)