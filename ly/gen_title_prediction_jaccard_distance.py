import pandas as pd
import numpy as np
import re
import os
from scipy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
import path_file

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
vec_txt = path_file.vec_txt
title_prediction_jaccard_distance_txt = path_file.title_prediction_jaccard_distance_txt

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

# with open(os.path.expanduser('~/Disk/TencentEmbedding/Tencent_AILab_ChineseEmbedding.txt'),'r',encoding='utf-8') as f:
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
#         # try:
#         #     vector = list(map(lambda x: float(x), vectorlist[-200:]))  # 对词向量进行处理
#         # except ValueError:
#         #     print(m)
#         #     vector = list(map(lambda x: float(x), vectorlist[2:]))  # 对词向量进行处理
#         #     vectorlist[0] = vectorlist[0] + vectorlist[1]
#
#         vector = list(map(lambda x: float(x), vectorlist[-200:]))  # 对词向量进行处理
#         vec = np.array(vector)  # 将列表转化为array
#         vecdic[''.join(vectorlist[:-200])] = vec

##########构造的特征###########
title_prediction_distance = pd.DataFrame()

pattern = re.compile(u'([\u4e00-\u9fff]+)')

import thulac
thu1 = thulac.thulac()  #默认模式

sentence_to_vec_dict = {}

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

def jaccard_distance(s1,s2):
    def add_space(s):
        return ' '.join(list(s))

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

title_tag_distances = []
title_is_all_zhs = []

title_prediction_jaccard_distance_means = []
title_prediction_jaccard_distance_stds = []
title_prediction_jaccard_distance_mins = []
title_prediction_jaccard_distance_maxs = []
title_prediction_jaccard_distance_w_means = []

title_max_prediction_jaccard_distances = []

data['title_digit_len'] = data['title'].apply(lambda x: sum(c.isdigit() for c in x))
data['title_alpha_len'] = data['title'].apply(lambda x: len(re.findall('[a-zA-Z]',x)))

for i,row in data.iterrows():
    if i%1000 == 0:
        print(i)
    # s：query_prediction字符串
    s = str(row['query_prediction'])
    s = s.replace('{', '')
    s = s.replace('}', '')

    title = str(row['title'])

    title_is_all_zh = 1
    if row['title_digit_len'] > 0 or row['title_alpha_len'] > 0:
        title_is_all_zh = 0

    title_vec = sentence_vec(title,title_is_all_zh)

    tag = str(row['tag'])
    tag_vec = sentence_vec(tag,title_is_all_zh)

    title_tag_distance = distance(title_vec,tag_vec)

    # max_pre_title和max_pre_title_score记录预测词组中最有可能的词组以及它的概率
    max_pre_title = ''
    max_pre_title_score = 0

    # s_list：[车辆行驶证": "0.018, 车辆违章查询": "0.242...]
    s_list = s.split('", "')

    all_pre = []
    all_jaccard_distance = []
    pre_dict = {}

    for item in s_list:
        if len(item) == 0:
            all_pre.append(0)
            all_jaccard_distance.append(1)
            continue

        # item_list举例:[车辆行驶证, 0.018]
        item_list = [t.replace('"', '') for t in item.split('": "')]
        all_pre.append(float(item_list[-1]))
        all_jaccard_distance.append(jaccard_distance(title,str(item_list[0])))
        pre_dict[str(item_list[0])] = float(item_list[-1])

        if float(item_list[-1]) > max_pre_title_score:
            max_pre_title_score = float(item_list[-1])
            max_pre_title = str(item_list[0])

    ##############################################
    ##pre feat
    ##############################################
    # 对概率统计
    pre_mean = np.mean(all_pre)

    title_prediction_jaccard_distance_mean = np.mean(all_jaccard_distance)
    title_prediction_jaccard_distance_std = np.std(all_jaccard_distance)
    title_prediction_jaccard_distance_min = np.min(all_jaccard_distance)
    title_prediction_jaccard_distance_max = np.max(all_jaccard_distance)
    title_prediction_jaccard_distance_w_mean = 1
    if pre_mean > 0:
        title_prediction_jaccard_distance_w_mean = np.sum(np.array(all_jaccard_distance) * np.array(all_pre)) / np.sum(all_pre)

    title_max_prediction_jaccard_distance = 1
    if pre_mean > 0:
        # title_max_prediction_jaccard_distance = distance(sentence_vec(max_pre_title,title_is_all_zh), title_vec)
        title_max_prediction_jaccard_distance = jaccard_distance(max_pre_title, title)


    title_is_all_zhs.append(title_is_all_zh)
    if not title_is_all_zh:
        title_tag_distances.append(1)
    else:
        title_tag_distances.append(title_tag_distance)

    title_prediction_jaccard_distance_means.append(title_prediction_jaccard_distance_mean)
    title_prediction_jaccard_distance_stds.append(title_prediction_jaccard_distance_std)
    title_prediction_jaccard_distance_mins.append(title_prediction_jaccard_distance_min)
    title_prediction_jaccard_distance_maxs.append(title_prediction_jaccard_distance_max)
    title_prediction_jaccard_distance_w_means.append(title_prediction_jaccard_distance_w_mean)

    title_max_prediction_jaccard_distances.append(title_max_prediction_jaccard_distance)

title_prediction_distance['title_is_all_zh'] = title_is_all_zhs
title_prediction_distance['title_tag_distance'] = title_tag_distances

title_prediction_distance['title_prediction_jaccard_distance_mean'] = title_prediction_jaccard_distance_means
title_prediction_distance['title_prediction_jaccard_distance_std'] = title_prediction_jaccard_distance_stds
title_prediction_distance['title_prediction_jaccard_distance_min'] = title_prediction_jaccard_distance_mins
title_prediction_distance['title_prediction_jaccard_distance_max'] = title_prediction_jaccard_distance_maxs
title_prediction_distance['title_prediction_jaccard_distance_w_mean'] = title_prediction_jaccard_distance_w_means

title_prediction_distance['title_max_prediction_jaccard_distance'] = title_max_prediction_jaccard_distances

print (title_prediction_distance.shape)


title_prediction_distance.to_csv(title_prediction_jaccard_distance_txt, sep='\t', index=False)