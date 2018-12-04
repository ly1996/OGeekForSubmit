import pandas as pd
import numpy as np
from sklearn import preprocessing
import lightgbm as lgb
import gc
import re
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, f1_score
from time import ctime
import os
# 数据路径
import path_file
import pandas as pd
import numpy as np
import pickle, os, jieba, time, gc, re
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import lightgbm as lgb
import datetime
import warnings
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

train_txt = path_file.train_txt
test_txt = path_file.test_txt
print(test_txt)
val_txt = path_file.val_txt

train_data = pd.read_table(train_txt,
                           names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                           quoting=3).astype(str)
val_data = pd.read_table(val_txt,
                         names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                         quoting=3).astype(str)
test_data = pd.read_table(test_txt,
                          names=['prefix', 'query_prediction', 'title', 'tag'], header=None, encoding='utf-8',
                          quoting=3).astype(str)
# 去噪，只有一条数据
train_data = train_data[train_data['label'].isin(['0', '1'])]

# 统一赋值test_data的label
test_data['label'] = -1

len_train = train_data.shape[0]
len_val = val_data.shape[0]
len_test = test_data.shape[0]
print("len_train", len_train)
print("len_val", len_val)
print("len_test", len_test)

# 连接在一起便于统一处理
data = pd.concat([train_data, val_data, test_data], ignore_index=True)
data['label'] = data['label'].apply(lambda x: int(x))

sentence_to_vec_dict = {}


def sentence_cut(s):
    s = str(s).lower()
    if s in sentence_to_vec_dict:
        return sentence_to_vec_dict[s]
    list_s = ' '.join(jieba.cut(s))
    sentence_to_vec_dict[s] = list_s
    return list_s


data['prefix_list'] = data['prefix'].apply(sentence_cut)
data['title_list'] = data['title'].apply(sentence_cut)

# 计算prefix,title的奇异值分解向量乘积做相似性度量
data['prefix_list'].fillna(' ', inplace=True)
data['title_list'].fillna(' ', inplace=True)
print(data[['prefix_list', 'title_list']].info())
prefix_vec = data['prefix_list'].tolist()
title_vec = data['title_list'].tolist()
corpus = prefix_vec + title_vec

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
tfidf.fit(corpus)
print('fit done')
prefix_corpus_tfidf = tfidf.transform(data['prefix_list'].values)
print('prefix_corpus_tfidf done')
title_corpus_tfidf = tfidf.transform(data['title_list'].values)
print('title_corpus_tfidf done')

corpus_tfidf = tfidf.transform(corpus)

# tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
svd = TruncatedSVD(30, algorithm='arpack', random_state=2018)
# lsa = make_pipeline(tfidf, normalizer, svd)

svd.fit(corpus_tfidf)
print('fit done')
title_vec = svd.transform(title_corpus_tfidf)
print('title_vec done')
prefix_vec = svd.transform(prefix_corpus_tfidf)
print('prefix_vec done')
# print(prefix_vec[:5])

tfidf_feat = pd.DataFrame()

tfidf_feat['title_prefix_cosine_distance_tfidf'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                   np.nan_to_num(title_vec))]
print('title_prefix_cosine_distance done')
tfidf_feat['title_prefix_cityblock_distance_tfidf'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                         np.nan_to_num(title_vec))]
print('title_prefix_cityblock_distance done')
tfidf_feat['title_prefix_jaccard_distance_tfidf'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                     np.nan_to_num(title_vec))]
print('title_prefix_jaccard_distance done')
tfidf_feat['title_prefix_canberra_distance_tfidf'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                       np.nan_to_num(title_vec))]
print('title_prefix_canberra_distance done')
tfidf_feat['title_prefix_euclidean_distance_tfidf'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                         np.nan_to_num(title_vec))]
print('title_prefix_euclidean_distance done')
tfidf_feat['title_prefix_minkowski_distance_tfidf'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                            np.nan_to_num(title_vec))]
print('title_prefix_minkowski_distance done')
tfidf_feat['title_prefix_braycurtis_distance_tfidf'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(prefix_vec),
                                                                                           np.nan_to_num(title_vec))]

dot_sim = []
consin_sim = []
for i in range(title_vec.shape[0]):
    dot_sim.append(np.dot(title_vec[i], prefix_vec[i].T))
    consin_sim.append(cosine_similarity([title_vec[i], prefix_vec[i]])[0][1])
tfidf_feat['dot_sim_tfidf'] = dot_sim
tfidf_feat['consin_sim_tfidf'] = consin_sim

## 语义向量交互，相应位置相减取绝对值,点乘
search_query1 = np.abs(prefix_vec - title_vec)
search_query2 = prefix_vec * title_vec
tfidf_feat['svd1_sim_mean_tfidf'] = search_query1.mean(1)
tfidf_feat['svd1_sim_sum_tfidf'] = search_query1.sum(1)
tfidf_feat['svd2_sim_mean_tfidf'] = search_query2.mean(1)
tfidf_feat['svd2_sim_sum_tfidf'] = search_query2.sum(1)

prefix_title_tfidf_txt = "../DataSets/feat_data/prefix_title_tfidf_feat.txt"
tfidf_feat.to_csv(prefix_title_tfidf_txt, index=False, sep='\t')