import pandas as pd
import numpy as np
import re
import os
from scipy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from time import ctime
import codecs, json
from gensim.models import KeyedVectors
import path_file
import gc
import w2v_util

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
dict_txt = path_file.dict_new_txt
title_tag_word2vec_distance_txt = path_file.title_tag_word2vec_distance_new_txt

# print('Start load sentence_to_vec_dict： %s' % (ctime()))
# file = open(dict_txt, 'r')
# js = file.read()
# sentence_to_vec_dict = json.loads(js)
# file.close()
# print('Endt load sentence_to_vec_dict： %s' % (ctime()))

sentence_to_vec_dict = {}

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
title_tag_distance = pd.DataFrame()

def sentence_vec(str):
    if str in sentence_to_vec_dict:
        return np.array(sentence_to_vec_dict[str])
    vec = w2v_util.sentence_vec(str)
    sentence_to_vec_dict[str] = vec.tolist()
    return vec

def distance(v1,v2):
    #余弦距离
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

tag = data.groupby(['tag'],as_index=False)['tag'].agg({'cnt':'count'})
print ("tag array len",len(tag))

tags = []
tagToVec = {}
tagToDistances = {}
tagToDistancesSub = {}
tagToDistancesRate = {}
for i,row in tag.iterrows():
    tag_name = str(row['tag'])
    tags.append(tag_name)
    tag_vec = sentence_vec(tag_name)
    tagToVec[tag_name] = tag_vec
    tagToDistances[tag_name] = []
    tagToDistancesSub[tag_name] = []
    tagToDistancesRate[tag_name] = []

title_tag_word2vec_distances = []

title_tag_distance_means = []
title_tag_distance_mins = []
title_tag_distance_maxs = []
title_tag_distance_stds = []

title_tag_distance_sub_means = []
title_tag_distance_sub_mins = []
title_tag_distance_sub_maxs = []

title_tag_distance_mean_rates = []
title_tag_distance_min_rates = []
title_tag_distance_max_rates = []

for i,row in data.iterrows():
    if i%1000 == 0:
        print(i)

    title = str(row['title'])
    tag = str(row['tag'])

    title_vec = sentence_vec(title)
    tag_vec = tagToVec[tag]
    tag_distance = distance(title_vec, tag_vec)
    distances = []

    for t in tags:
        t_vec = tagToVec[t]
        d = distance(title_vec,t_vec)
        distances.append(d)
        tagToDistances[t].append(d)
        tagToDistancesSub[t].append(tag_distance - d)
        tagToDistancesRate[t].append(tag_distance / d)

    title_tag_distance_mean = np.mean(distances)
    title_tag_distance_min = np.min(distances)
    title_tag_distance_max = np.max(distances)
    title_tag_distance_std = np.std(distances)

    title_tag_distance_sub_mean = tag_distance - title_tag_distance_mean
    title_tag_distance_sub_min = tag_distance - title_tag_distance_min
    title_tag_distance_sub_max = tag_distance - title_tag_distance_max

    title_tag_distance_mean_rate = tag_distance / title_tag_distance_mean
    title_tag_distance_min_rate = tag_distance / title_tag_distance_min
    title_tag_distance_max_rate = tag_distance / title_tag_distance_max

    title_tag_word2vec_distances.append(tag_distance)
    title_tag_distance_means.append(title_tag_distance_mean)
    title_tag_distance_mins.append(title_tag_distance_min)
    title_tag_distance_maxs.append(title_tag_distance_max)
    title_tag_distance_stds.append(title_tag_distance_std)

    title_tag_distance_sub_means.append(title_tag_distance_sub_mean)
    title_tag_distance_sub_mins.append(title_tag_distance_sub_min)
    title_tag_distance_sub_maxs.append(title_tag_distance_sub_max)

    title_tag_distance_mean_rates.append(title_tag_distance_mean_rate)
    title_tag_distance_min_rates.append(title_tag_distance_min_rate)
    title_tag_distance_max_rates.append(title_tag_distance_max_rate)

title_tag_distance['title_tag_word2vec_distance'] = title_tag_word2vec_distances

title_tag_distance['title_tag_distance_mean'] = title_tag_distance_means
title_tag_distance['title_tag_distance_min'] = title_tag_distance_mins
title_tag_distance['title_tag_distance_max'] = title_tag_distance_maxs
title_tag_distance['title_tag_distance_std'] = title_tag_distance_stds

title_tag_distance['title_tag_distance_sub_mean'] = title_tag_distance_sub_means
title_tag_distance['title_tag_distance_sub_min'] = title_tag_distance_sub_mins
title_tag_distance['title_tag_distance_sub_max'] = title_tag_distance_sub_maxs

title_tag_distance['title_tag_distance_mean_rate'] = title_tag_distance_mean_rates
title_tag_distance['title_tag_distance_min_rate'] = title_tag_distance_min_rates
title_tag_distance['title_tag_distance_max_rate'] = title_tag_distance_max_rates

for tag in tags:
    title_tag_distance[tag + 'distance'] = tagToDistances[tag]
    title_tag_distance[tag + 'distance_sub'] = tagToDistancesSub[tag]
    title_tag_distance[tag + 'distance_rate'] = tagToDistancesRate[tag]

print (title_tag_distance.shape)
title_tag_distance.to_csv(title_tag_word2vec_distance_txt, sep='\t', index=False)



