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

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
dict_txt = path_file.dict_txt
vec_txt = path_file.vec_txt
title_tag_word2vec_distance_txt = path_file.title_tag_word2vec_distance_txt

print('Start load sentence_to_vec_dict： %s' % (ctime()))
file = open(dict_txt, 'r')
js = file.read()
sentence_to_vec_dict = json.loads(js)
file.close()
print('Endt load sentence_to_vec_dict： %s' % (ctime()))

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

# begin = 0
# end = 10000
# data = data.iloc[begin:end]
# print(begin,' to ', end)

#构造分词和向量之间的字典
vecdic = KeyedVectors.load_word2vec_format(vec_txt, binary=False)
vec_len = len(vecdic['重庆'].tolist())
print (vec_len)

# with open(os.path.expanduser('~/Disk/TencentEmbedding/Tencent_AILab_ChineseEmbedding0.txt'),'r',encoding='utf-8') as f:
#     f.readline()#第一行为词汇数和向量维度，在这里不予展示
#     # f.readline()
#
#     # m=f.readline()#读取第三个词
#     i = 0
#     while True:
#         m = f.readline()
#         i = i + 1
#         if i % 10000 == 0:
#             print(i)
#         if not m:
#             break
#         vectorlist = m.split()  # 切分一行，分为词汇和词向量
#
#         vector = list(map(lambda x: float(x), vectorlist[-200:]))  # 对词向量进行处理
#         vec = np.array(vector)  # 将列表转化为array
#         vecdic[''.join(vectorlist[:-200])] = vec

##########构造的特征###########
title_tag_distance = pd.DataFrame()

pattern = re.compile(u'([\u4e00-\u9fff]+)')

import thulac
thu1 = thulac.thulac()  #默认模式

def sentence_vec(str,title_is_all_zh):
    if str in sentence_to_vec_dict:
        # print ("same")
        return np.array(sentence_to_vec_dict[str])
    if not title_is_all_zh:
        return np.ones(vec_len)
    cut = thu1.cut(str, text=False)
    sentence = np.zeros(vec_len)
    for item in cut:
        word = item[0]
        try:
            word_vec = vecdic[word]
        except KeyError:
            continue
        else:
            sentence += word_vec
    sentence /= len(cut)
    sentence_to_vec_dict[str] = sentence.tolist()
    return sentence

def distance(v1,v2):
    #余弦距离
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

tag = data.groupby(['tag'],as_index=False)['tag'].agg({'cnt':'count'})
print (len(tag))

tags = []
tagToVec = {}
tagToDistances = {}
tagToDistancesSub = {}
tagToDistancesRate = {}
for i,row in tag.iterrows():
    tag_name = str(row['tag'])
    tags.append(tag_name)
    tag_vec = sentence_vec(tag_name,1)
    tagToVec[tag_name] = tag_vec
    tagToDistances[tag_name] = []
    tagToDistancesSub[tag_name] = []
    tagToDistancesRate[tag_name] = []

data['title_digit_len'] = data['title'].apply(lambda x: sum(c.isdigit() for c in x))
data['title_alpha_len'] = data['title'].apply(lambda x: len(re.findall('[a-zA-Z]',x)))

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

    title_is_all_zh = 1
    if row['title_digit_len'] > 0 or row['title_alpha_len'] > 0:
        title_is_all_zh = 0

    title_vec = sentence_vec(title, title_is_all_zh)

    tag_vec = tagToVec[tag]
    tag_distance = distance(title_vec,tag_vec)
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

    title_tag_distance_mean_rate = tag_distance/title_tag_distance_mean
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

# title_tag_distance.to_csv('data/feat_data/title_tag_word2vec_distance_'+str(begin)+'_'+str(end)+'.csv',sep='\t',index=False)
print (title_tag_distance.shape)
title_tag_distance.to_csv(title_tag_word2vec_distance_txt, sep='\t', index=False)




