import pandas as pd
import numpy as np
import re
import gc
from gensim.models import KeyedVectors
import path_file
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import w2v_util
import json

######################################

######################
# load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
# vec_txt = path_file.vec_txt
nlp_feat_txt = path_file.nlp_feat_txt

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

# 连接在一起便于统一处理
data = pd.concat([train_data, val_data, test_data], ignore_index=True)
data['label'] = data['label'].apply(lambda x: int(x))

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

del train_data
del val_data
del test_data
gc.collect()

vec_len = w2v_util.vec_len

##############################


# 构造分词和向量之间的字典
# vecdic = KeyedVectors.load_word2vec_format(vec_txt, binary=False)
# vec_len = len(vecdic['重庆'].tolist())
# print (vec_len)

#####################################

pattern = re.compile(u'([\u4e00-\u9fff]+)')

# import thulac
#
# thu1 = thulac.thulac()  # 默认模式

sentence_to_vec_dict = {}

##############################
# def sentence_vec(s):
#     s = str(s)
#     if s in sentence_to_vec_dict:
#         #         print ("same")
#         return np.array(sentence_to_vec_dict[s])
#
#     cut = thu1.cut(s, text=False)
#     #     print(cut)
#     sentence = np.zeros(vec_len)
#     for item in cut:
#         word = item[0]
#         try:
#             word_vec = vecdic[word]
#         except KeyError:
#             continue
#         else:
#             sentence += word_vec
#     sentence /= len(cut)
#     sentence_to_vec_dict[s] = sentence.tolist()
#     return sentence
def sentence_vec(str):
    if str in sentence_to_vec_dict:
        return np.array(sentence_to_vec_dict[str])
    vec = w2v_util.sentence_vec(str)
    sentence_to_vec_dict[str] = vec.tolist()
    return vec

####################################

def query_prediction2vec(s):
    mean_vector = np.zeros(vec_len)
    w_mean_vector = np.zeros(vec_len)

    cnt = 0
    for query_item, query_ratio in s.items():
        tmp_vecotr = sentence_vec(str(query_item))
        mean_vector += tmp_vecotr
        w_mean_vector += float(query_ratio) * tmp_vecotr
        cnt = cnt + 1
    mean_vector = mean_vector / cnt
    return mean_vector, w_mean_vector

    # s = s.replace('{', '')
    # s = s.replace('}', '')
    # s_list = s.split('", "')
    # for item in s_list:
    #     if len(item) == 0:
    #         break
    #     item_list = [t.replace('"', '') for t in item.split('": "')]
    #     pre_score = float(item_list[-1])
    #     pre_title = str(item_list[0])
    #     #         print(pre_title)
    #     tmp_vecotr = sentence_vec(pre_title)
    #     mean_vector += tmp_vecotr
    #     w_mean_vector += pre_score * tmp_vecotr
    #
    # mean_vector = mean_vector / len(s_list)
    # return mean_vector, w_mean_vector


####################################

def loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")

data["query_prediction"] = data["query_prediction"].apply(loads)

######################
# sentence_to_vec_dict = {}
prefix_vectors = np.zeros((data.shape[0], vec_len))
for i, q in tqdm(enumerate(data.prefix.values)):
    prefix_vectors[i, :] = sentence_vec(q)

prefix_vectors = np.float32(prefix_vectors)
########################
# sentence_to_vec_dict = {}
title_vectors = np.zeros((data.shape[0], vec_len))
for i, q in tqdm(enumerate(data.title.values)):
    title_vectors[i, :] = sentence_vec(q)

title_vectors = np.float32(title_vectors)
############################

# del data
# del sentence_to_vec_dict
gc.collect()

#######################

nlp_feat = pd.DataFrame()

# nlp_feat['title_kur_ybb'] = [kurtosis(x) for x in np.nan_to_num(title_vectors)]
#
# nlp_feat.to_csv(nlp_feat_txt, sep='\t', index=False)

########################

nlp_feat['title_prefix_cosine_distance_ybb'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                               np.nan_to_num(title_vectors))]
print('title_prefix_cosine_distance done')
nlp_feat['title_prefix_cityblock_distance_ybb'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                     np.nan_to_num(title_vectors))]
print('title_prefix_cityblock_distance done')
nlp_feat['title_prefix_jaccard_distance_ybb'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                 np.nan_to_num(title_vectors))]
print('title_prefix_jaccard_distance done')
nlp_feat['title_prefix_canberra_distance_ybb'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                   np.nan_to_num(title_vectors))]
print('title_prefix_canberra_distance done')
nlp_feat['title_prefix_euclidean_distance_ybb'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                     np.nan_to_num(title_vectors))]
print('title_prefix_euclidean_distance done')
nlp_feat['title_prefix_minkowski_distance_ybb'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                        np.nan_to_num(title_vectors))]
print('title_prefix_minkowski_distance done')
nlp_feat['title_prefix_braycurtis_distance_ybb'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(prefix_vectors),
                                                                                       np.nan_to_num(title_vectors))]
print('title_prefix_braycurtis_distance done')
nlp_feat['prefix_skew_ybb'] = [skew(x) for x in np.nan_to_num(prefix_vectors)]
nlp_feat['title_skew_ybb'] = [skew(x) for x in np.nan_to_num(title_vectors)]
nlp_feat['prefix_kur_ybb'] = [kurtosis(x) for x in np.nan_to_num(prefix_vectors)]
nlp_feat['title_kur_ybb'] = [kurtosis(x) for x in np.nan_to_num(title_vectors)]

del prefix_vectors
# del title_vectors
gc.collect()
########################

query_prediction_mean2vec = np.zeros((data.shape[0], vec_len))
query_prediction_wmean2vec = np.zeros((data.shape[0], vec_len))
for i, q in tqdm(enumerate(data.query_prediction.values)):
    if i % 1000 == 0:
        print (i)
    query_prediction_mean2vec[i, :], query_prediction_wmean2vec[i, :] = query_prediction2vec(q)

# del data
# gc.collect()
query_prediction_mean2vec = np.float32(query_prediction_mean2vec)
query_prediction_wmean2vec = np.float32(query_prediction_wmean2vec)

nlp_feat['title_premean_cosine_distance_ybb'] = [cosine(x, y) for (x, y) in
                                                 zip(np.nan_to_num(query_prediction_mean2vec),
                                                     np.nan_to_num(title_vectors))]
print('title_premean_cosine_distance done')
nlp_feat['title_premean_cityblock_distance_ybb'] = [cityblock(x, y) for (x, y) in
                                                    zip(np.nan_to_num(query_prediction_mean2vec),
                                                        np.nan_to_num(title_vectors))]
print('title_premean_cityblock_distance done')
nlp_feat['title_premean_jaccard_distance_ybb'] = [jaccard(x, y) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_mean2vec),
                                                      np.nan_to_num(title_vectors))]
print('title_premean_jaccard_distance done')
nlp_feat['title_premean_canberra_distance_ybb'] = [canberra(x, y) for (x, y) in
                                                   zip(np.nan_to_num(query_prediction_mean2vec),
                                                       np.nan_to_num(title_vectors))]
print('title_premean_canberra_distance done')
nlp_feat['title_premean_euclidean_distance_ybb'] = [euclidean(x, y) for (x, y) in
                                                    zip(np.nan_to_num(query_prediction_mean2vec),
                                                        np.nan_to_num(title_vectors))]
print('title_premean_euclidean_distance done')
nlp_feat['title_premean_minkowski_distance_ybb'] = [minkowski(x, y, 3) for (x, y) in
                                                    zip(np.nan_to_num(query_prediction_mean2vec),
                                                        np.nan_to_num(title_vectors))]
print('title_premean_minkowski_distance done')
nlp_feat['title_premean_braycurtis_distance_ybb'] = [braycurtis(x, y) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_mean2vec),
                                                         np.nan_to_num(title_vectors))]
print('title_premean_braycurtis_distance done')
###########################

nlp_feat['title_prewmean_cosine_distance_ybb'] = [cosine(x, y) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_wmean2vec),
                                                      np.nan_to_num(title_vectors))]
print('title_prewmean_cosine_distance done')
nlp_feat['title_prewmean_cityblock_distance_ybb'] = [cityblock(x, y) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_wmean2vec),
                                                         np.nan_to_num(title_vectors))]
print('title_prewmean_cityblock_distance done')
nlp_feat['title_prewmean_jaccard_distance_ybb'] = [jaccard(x, y) for (x, y) in
                                                   zip(np.nan_to_num(query_prediction_wmean2vec),
                                                       np.nan_to_num(title_vectors))]
print('title_prewmean_jaccard_distance done')
nlp_feat['title_prewmean_canberra_distance_ybb'] = [canberra(x, y) for (x, y) in
                                                    zip(np.nan_to_num(query_prediction_wmean2vec),
                                                        np.nan_to_num(title_vectors))]
print('title_prewmean_canberra_distance done')
nlp_feat['title_prewmean_euclidean_distance_ybb'] = [euclidean(x, y) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_wmean2vec),
                                                         np.nan_to_num(title_vectors))]
print('title_prewmean_euclidean_distance done')
nlp_feat['title_prewmean_minkowski_distance_ybb'] = [minkowski(x, y, 3) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_wmean2vec),
                                                         np.nan_to_num(title_vectors))]
print('title_prewmean_minkowski_distance done')
nlp_feat['title_prewmean_braycurtis_distance_ybb'] = [braycurtis(x, y) for (x, y) in
                                                      zip(np.nan_to_num(query_prediction_wmean2vec),
                                                          np.nan_to_num(title_vectors))]
print('title_prewmean_braycurtis_distance done')

del title_vectors
gc.collect()
###############################
prefix_vectors = np.zeros((data.shape[0], vec_len))
for i, q in tqdm(enumerate(data.prefix.values)):
    prefix_vectors[i, :] = sentence_vec(q)

prefix_vectors = np.float32(prefix_vectors)

nlp_feat['prefix_premean_cosine_distance_ybb'] = [cosine(x, y) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_mean2vec),
                                                      np.nan_to_num(prefix_vectors))]
print('prefix_premean_cosine_distance done')
nlp_feat['prefix_premean_cityblock_distance_ybb'] = [cityblock(x, y) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_mean2vec),
                                                         np.nan_to_num(prefix_vectors))]
print('prefix_premean_cityblock_distance done')
nlp_feat['prefix_premean_jaccard_distance_ybb'] = [jaccard(x, y) for (x, y) in
                                                   zip(np.nan_to_num(query_prediction_mean2vec),
                                                       np.nan_to_num(prefix_vectors))]
print('prefix_premean_jaccard_distance done')
nlp_feat['prefix_premean_canberra_distance_ybb'] = [canberra(x, y) for (x, y) in
                                                    zip(np.nan_to_num(query_prediction_mean2vec),
                                                        np.nan_to_num(prefix_vectors))]
print('prefix_premean_canberra_distance done')
nlp_feat['prefix_premean_euclidean_distance_ybb'] = [euclidean(x, y) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_mean2vec),
                                                         np.nan_to_num(prefix_vectors))]
print('prefix_premean_euclidean_distance done')
nlp_feat['prefix_premean_minkowski_distance_ybb'] = [minkowski(x, y, 3) for (x, y) in
                                                     zip(np.nan_to_num(query_prediction_mean2vec),
                                                         np.nan_to_num(prefix_vectors))]
print('prefix_premean_minkowski_distance done')
nlp_feat['prefix_premean_braycurtis_distance_ybb'] = [braycurtis(x, y) for (x, y) in
                                                      zip(np.nan_to_num(query_prediction_mean2vec),
                                                          np.nan_to_num(prefix_vectors))]
print('prefix_premean_braycurtis_distance done')

######################################

nlp_feat['prefix_prewmean_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(query_prediction_wmean2vec),
                                                                              np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_cosine_distance done')
nlp_feat['prefix_prewmean_cityblock_distance'] = [cityblock(x, y) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_wmean2vec),
                                                      np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_cityblock_distance done')
nlp_feat['prefix_prewmean_jaccard_distance'] = [jaccard(x, y) for (x, y) in
                                                zip(np.nan_to_num(query_prediction_wmean2vec),
                                                    np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_jaccard_distance done')
nlp_feat['prefix_prewmean_canberra_distance'] = [canberra(x, y) for (x, y) in
                                                 zip(np.nan_to_num(query_prediction_wmean2vec),
                                                     np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_canberra_distance done')
nlp_feat['prefix_prewmean_euclidean_distance'] = [euclidean(x, y) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_wmean2vec),
                                                      np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_euclidean_distance done')
nlp_feat['prefix_prewmean_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in
                                                  zip(np.nan_to_num(query_prediction_wmean2vec),
                                                      np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_minkowski_distance done')
nlp_feat['prefix_prewmean_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in
                                                   zip(np.nan_to_num(query_prediction_wmean2vec),
                                                       np.nan_to_num(prefix_vectors))]
print('prefix_prewmean_braycurtis_distance done')

del prefix_vectors
gc.collect()
#########################################

nlp_feat['premean_skew_ybb'] = [skew(x) for x in np.nan_to_num(query_prediction_mean2vec)]
nlp_feat['premean_kur_ybb'] = [kurtosis(x) for x in np.nan_to_num(query_prediction_mean2vec)]

nlp_feat['prewmean_skew_ybb'] = [skew(x) for x in np.nan_to_num(query_prediction_wmean2vec)]
nlp_feat['prewmean_kur_ybb'] = [kurtosis(x) for x in np.nan_to_num(query_prediction_wmean2vec)]

#########################################

nlp_feat.to_csv(nlp_feat_txt, sep='\t', index=False)