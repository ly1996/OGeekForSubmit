import pandas as pd
import numpy as np

import re
import path_file
import json
import jieba

######################
#load data
######################

train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
query_prediction_feature_txt = path_file.query_prediction_feature_new_txt

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

def loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")


def get_rp_prefix_in_title(prefix, title, mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))
    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))

    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall, precision, acc]


def get_ngram_rp_prefix_in_title(prefix, title, mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率（1-2gram）"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))
    prefix_2gram = []
    for i in range(len(prefix) - 1):
        prefix_2gram.append(prefix[i] + prefix[i + 1])
    prefix.extend(prefix_2gram)

    title_2gram = []
    for i in range(len(title) - 1):
        title_2gram.append(title[i] + title[i + 1])
    title.extend(title_2gram)

    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))

    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall, precision, acc]

data["query_prediction"] = data["query_prediction"].apply(loads)

# begin = 0
# end = 1000
# data = data.iloc[begin:end]
# print(begin,' to ', end)

##########构造的特征###########
query_prediction_feat = pd.DataFrame()
pre_means = []
pre_stds = []
pre_maxs = []
pre_mins = []
pre_cnts = []
pre_len_means = []
pre_len_stds = []
pre_len_maxs = []
pre_len_mins = []
title_is_in_pres = []
title_in_pre_scores = []
in_query_bigs = []
title_in_pre_score_sub_maxs = []
title_in_pre_score_sub_mins = []
title_in_pre_score_sub_means = []
title_len_sub_maxs = []
title_len_sub_mins = []
title_len_sub_means = []
zh_pre_title_common_words_cnt_means = []
zh_pre_title_common_words_cnt_stds = []
zh_pre_title_common_words_cnt_maxs = []
zh_pre_title_common_words_cnt_mins = []
pre_title_common_words_cnt_means = []
pre_title_common_words_cnt_stds = []
pre_title_common_words_cnt_maxs = []
pre_title_common_words_cnt_mins = []
pre_title_common_words_cnt_mean_rates = []
pre_title_common_words_cnt_max_rates = []
pre_title_common_words_cnt_min_rates = []
max_pre_title_common_words_cnts = []
zh_max_pre_title_common_words_cnts = []
max_pre_title_common_words_cnt_rates = []

pre_len_w_means = []
title_len_sub_w_means = []
prefix_is_in_pres = []
prefix_in_pre_scores = []
prefix_in_pre_score_sub_maxs = []
prefix_in_pre_score_sub_mins = []
prefix_in_pre_score_sub_means = []
prefix_len_sub_maxs = []
prefix_len_sub_mins = []
prefix_len_sub_means = []
prefix_len_sub_w_means = []
prefix_len_rate_maxs = []
prefix_len_rate_mins = []
prefix_len_rate_means = []
prefix_len_rate_w_means = []
max_pre_prefix_common_words_cnt_rates = []
zh_max_pre_prefix_common_words_cnt_rates = []

pre_len_new_w_means = []
title_len_sub_new_w_means = []

#添加：概率的比值
title_in_pre_score_max_rates = []
title_in_pre_score_min_rates = []
title_in_pre_score_mean_rates = []

#添加：长度的比值
title_len_max_rates = []
title_len_min_rates = []
title_len_mean_rates = []
title_len_w_mean_rates = []
title_len_new_w_mean_rates = []

#添加权重因素的共同字符
pre_title_common_words_cnt_w_means = []

#添加权重因素的共同字符的比值
pre_title_common_words_cnt_w_mean_rates = []

##common word rate in prediction##
pre_title_common_words_rate_means = []
pre_title_common_words_rate_maxs = []
pre_title_common_words_rate_mins = []
pre_title_common_words_rate_stds = []
pre_title_common_words_rate_w_means = []

zh_pre_title_common_words_rate_means = []
zh_pre_title_common_words_rate_maxs = []
zh_pre_title_common_words_rate_mins = []
zh_pre_title_common_words_rate_stds = []
# zh_pre_title_common_words_rate_w_means = []

max_pre_title_common_words_rates = []

query_t_recall_chars = []
query_t_precision_chars = []
query_t_acc_chars = []
query_t_recall_words = []
query_t_precision_words = []
query_t_acc_words = []
query_t_recall_char_ngrams = []
query_t_precision_char_ngrams = []
query_t_acc_char_ngrams = []
query_t_recall_word_ngrams = []
query_t_precision_word_ngrams = []
query_t_acc_word_ngrams = []

pattern = re.compile(u'([\u4e00-\u9fff]+)')

for i,row in data.iterrows():
    if i%1000 == 0:
        print(i)

    # s：query_prediction字符串
    s = row['query_prediction']

    title = str(row['title']).lower()
    prefix = str(row['prefix']).lower()
    zh_title = ''.join(pattern.findall(title))

    # max_pre_title和max_pre_title_score记录预测词组中最有可能的词组以及它的概率
    max_pre_title = ''
    max_pre_title_score = -0.1

    #all_pre：词组的概率
    all_pre = []
    #all_pre_len：词组的长度
    all_pre_len = []

    #词组和对应的概率
    pre_dict = {}
    zh_pre_dict = {}

    for query_item, query_ratio in s.items():
        if str(query_item).lower() in pre_dict:
            pre_dict[str(query_item).lower()] = pre_dict[str(query_item).lower()] + float(query_ratio)
        else:
            pre_dict[str(query_item).lower()] = float(query_ratio)
        zh_pre_dict[''.join(pattern.findall(str(query_item)))] = float(query_ratio)
        if pre_dict[str(query_item).lower()] > max_pre_title_score:
            max_pre_title_score = pre_dict[str(query_item).lower()]
            max_pre_title = str(query_item).lower()

    for query_item, query_ratio in pre_dict.items():
        all_pre.append(float(query_ratio))
        all_pre_len.append(len(str(query_item)))
    ##############################################
    ##pre feat
    ##############################################
    # pre_cnt预测的长度
    pre_cnt = len(all_pre)

    #对概率统计
    pre_mean = 0
    pre_std = 0
    pre_max = 0
    pre_min = 0

    #对长度进行统计
    pre_len_mean = 0
    pre_len_std = 0
    pre_len_max = 0
    pre_len_min = 0
    pre_len_w_mean = 0
    pre_len_new_w_mean = 0
    if pre_cnt > 0:
        pre_mean = np.mean(all_pre)
        pre_std = np.std(all_pre)
        pre_max = np.max(all_pre)
        pre_min = np.min(all_pre)

        pre_len_mean = np.mean(all_pre_len)
        pre_len_std = np.std(all_pre_len)
        pre_len_max = np.max(all_pre_len)
        pre_len_min = np.min(all_pre_len)
        pre_len_w_mean = np.sum(np.array(all_pre_len) * np.array(all_pre))
        pre_len_new_w_mean = np.sum(np.array(all_pre_len) * np.array(all_pre))/np.sum(all_pre)

    #pre_mean == 0说明为空{}
    #pre_cnt预测的长度
    # if pre_mean == 0:
    #     pre_cnt = 0
    # else:
    #     pre_cnt = len(all_pre)

    #####################################
    # title feat
    #####################################

    ##title pre##
    title_is_in_pre = 0
    title_in_pre_score = 0
    in_query_big = 0
    if title in pre_dict:
        title_is_in_pre = 1
        title_in_pre_score = pre_dict[title]
        if title_in_pre_score > 0.1:
            in_query_big = 1

    title_in_pre_score_sub_max = pre_max - title_in_pre_score
    title_in_pre_score_sub_min = pre_min - title_in_pre_score
    title_in_pre_score_sub_mean = pre_mean - title_in_pre_score

    #添加：概率的比值
    title_in_pre_score_max_rate = 0
    title_in_pre_score_min_rate = 0
    title_in_pre_score_mean_rate = 0

    if pre_cnt > 0:
        title_in_pre_score_max_rate = title_in_pre_score / pre_max
        title_in_pre_score_min_rate = title_in_pre_score / pre_min
        title_in_pre_score_mean_rate = title_in_pre_score / pre_mean

    ##title len##
    title_len = len(title)
    title_len_sub_max = pre_len_max - title_len
    title_len_sub_min = pre_len_min - title_len
    title_len_sub_mean = pre_len_mean - title_len
    title_len_sub_w_mean = pre_len_w_mean - title_len
    title_len_sub_new_w_mean = pre_len_new_w_mean - title_len

    #添加：长度的比值
    title_len_max_rate = pre_len_max / title_len
    title_len_min_rate = pre_len_min / title_len
    title_len_mean_rate = pre_len_mean / title_len
    title_len_w_mean_rate = pre_len_w_mean / title_len
    title_len_new_w_mean_rate = pre_len_new_w_mean / title_len

    ##title common words##
    zh_common_words_cnt = []
    common_words_cnt = []

    ##common word rate in prediction##
    zh_common_words_rate = []
    common_words_rate = []

    for key in zh_pre_dict:
        # print (key)
        zh_common_words_cnt.append(len(set(list(str(key))).intersection(set(list(str(zh_title))))))
        if len(str(key)) == 0:
            zh_common_words_rate.append(0)
        else:
            zh_common_words_rate.append(len(set(list(str(key))).intersection(set(list(str(zh_title)))))/len(str(key)))

    for key in pre_dict:
        common_words_cnt.append(len(set(list(str(key))).intersection(set(list(str(title))))))
        common_words_rate.append(len(set(list(str(key))).intersection(set(list(str(title))))) / len(str(key)))

    zh_pre_title_common_words_cnt_mean = 0
    zh_pre_title_common_words_cnt_std = 0
    zh_pre_title_common_words_cnt_max = 0
    zh_pre_title_common_words_cnt_min = 0
    pre_title_common_words_cnt_mean = 0
    pre_title_common_words_cnt_std = 0
    pre_title_common_words_cnt_max = 0
    pre_title_common_words_cnt_min = 0

    #添加权重因素的共同字符
    pre_title_common_words_cnt_w_mean = 0

    ##common word rate in prediction##
    pre_title_common_words_rate_mean = 0
    pre_title_common_words_rate_max = 0
    pre_title_common_words_rate_min = 0
    pre_title_common_words_rate_std = 0
    pre_title_common_words_rate_w_mean = 0

    zh_pre_title_common_words_rate_mean = 0
    zh_pre_title_common_words_rate_max = 0
    zh_pre_title_common_words_rate_min = 0
    zh_pre_title_common_words_rate_std = 0
    # zh_pre_title_common_words_rate_w_mean = 0

    if pre_cnt > 0:
        zh_pre_title_common_words_cnt_mean = np.mean(zh_common_words_cnt)
        zh_pre_title_common_words_cnt_std = np.std(zh_common_words_cnt)
        zh_pre_title_common_words_cnt_max = np.max(zh_common_words_cnt)
        zh_pre_title_common_words_cnt_min = np.min(zh_common_words_cnt)
        pre_title_common_words_cnt_mean = np.mean(common_words_cnt)
        pre_title_common_words_cnt_std = np.std(common_words_cnt)
        pre_title_common_words_cnt_max = np.max(common_words_cnt)
        pre_title_common_words_cnt_min = np.min(common_words_cnt)

        pre_title_common_words_cnt_w_mean = np.sum(np.array(common_words_cnt) * np.array(all_pre))/np.sum(all_pre)

        ##common word rate in prediction##
        pre_title_common_words_rate_mean = np.mean(common_words_rate)
        pre_title_common_words_rate_max = np.max(common_words_rate)
        pre_title_common_words_rate_min = np.min(common_words_rate)
        pre_title_common_words_rate_std = np.std(common_words_rate)

        pre_title_common_words_rate_w_mean = np.sum(np.array(common_words_rate) * np.array(all_pre))/np.sum(all_pre)

        zh_pre_title_common_words_rate_mean = np.mean(zh_common_words_rate)
        zh_pre_title_common_words_rate_max = np.max(zh_common_words_rate)
        zh_pre_title_common_words_rate_min = np.min(zh_common_words_rate)
        zh_pre_title_common_words_rate_std = np.std(zh_common_words_rate)

        # zh_pre_title_common_words_rate_w_mean = np.sum(np.array(zh_common_words_rate) * np.array(all_pre)) / np.sum(all_pre)

    pre_title_common_words_cnt_mean_rate = pre_title_common_words_cnt_mean / title_len
    pre_title_common_words_cnt_max_rate = pre_title_common_words_cnt_max / title_len
    pre_title_common_words_cnt_min_rate = pre_title_common_words_cnt_min / title_len

    #添加权重因素的共同字符的比值
    pre_title_common_words_cnt_w_mean_rate = pre_title_common_words_cnt_w_mean / title_len

    ##########max pre title feat#############
    zh_max_pre_title = ''.join(pattern.findall(str(max_pre_title)))

    max_pre_title_common_words_cnt = 0
    zh_max_pre_title_common_words_cnt = 0
    max_pre_title_common_words_cnt_rate = 0

    max_pre_title_common_words_rate = 0

    query_t_recall_char = 0
    query_t_precision_char = 0
    query_t_acc_char = 0
    query_t_recall_word = 0
    query_t_precision_word = 0
    query_t_acc_word = 0
    query_t_recall_char_ngram = 0
    query_t_precision_char_ngram = 0
    query_t_acc_char_ngram = 0
    query_t_recall_word_ngram = 0
    query_t_precision_word_ngram = 0
    query_t_acc_word_ngram = 0

    if pre_cnt > 0:
        max_pre_title_common_words_cnt = len(set(list(str(max_pre_title))).intersection(set(list(str(title)))))
        zh_max_pre_title_common_words_cnt = len(
            set(list(str(zh_max_pre_title))).intersection(set(list(str(zh_title)))))
        max_pre_title_common_words_cnt_rate = max_pre_title_common_words_cnt / title_len
        max_pre_title_common_words_rate = max_pre_title_common_words_cnt / len(str(max_pre_title))
        query_t_recall_char,query_t_precision_char,query_t_acc_char = get_rp_prefix_in_title(max_pre_title, title, mode='char')
        query_t_recall_word,query_t_precision_word,query_t_acc_word = get_rp_prefix_in_title(max_pre_title, title, mode='word')
        query_t_recall_char_ngram,query_t_precision_char_ngram,query_t_acc_char_ngram = get_ngram_rp_prefix_in_title(max_pre_title, title, mode='char')
        query_t_recall_word_ngram, query_t_precision_word_ngram, query_t_acc_word_ngram = get_ngram_rp_prefix_in_title(max_pre_title, title, mode='word')

    #####################################
    # prefix feat
    #####################################

    ##prefix pre##
    prefix_is_in_pre = 0
    prefix_in_pre_score = 0
    if prefix in pre_dict:
        prefix_is_in_pre = 1
        prefix_in_pre_score = pre_dict[prefix]

    prefix_in_pre_score_sub_max = pre_max - prefix_in_pre_score
    prefix_in_pre_score_sub_min = pre_min - prefix_in_pre_score
    prefix_in_pre_score_sub_mean = pre_mean - prefix_in_pre_score

    ##prefix len##
    prefix_len = len(prefix)
    prefix_len_sub_max = pre_len_max - prefix_len
    prefix_len_sub_min = pre_len_min - prefix_len
    prefix_len_sub_mean = pre_len_mean - prefix_len
    prefix_len_sub_w_mean = pre_len_w_mean - prefix_len

    prefix_len_rate_max = 0
    prefix_len_rate_min = 0
    prefix_len_rate_mean = 0
    prefix_len_rate_w_mean = 0

    if pre_cnt > 0:
        prefix_len_rate_max = prefix_len / pre_len_max
        prefix_len_rate_min = prefix_len / pre_len_min
        prefix_len_rate_mean = prefix_len / pre_len_mean
        prefix_len_rate_w_mean = prefix_len / pre_len_w_mean

    ##########max pre prefix feat#############
    zh_max_pre_title = ''.join(pattern.findall(str(max_pre_title)))
    max_pre_prefix_common_words_cnt_rate = 0
    zh_max_pre_prefix_common_words_cnt_rate = 0
    if pre_cnt > 0:
        if len(zh_max_pre_title) > 0:
            zh_max_pre_prefix_common_words_cnt_rate = prefix_len / len(zh_max_pre_title)
        max_pre_prefix_common_words_cnt_rate = prefix_len / len(max_pre_title)

    #######################################
    pre_means.append(pre_mean)
    pre_stds.append(pre_std)
    pre_maxs.append(pre_max)
    pre_mins.append(pre_min)
    pre_cnts.append(pre_cnt)
    pre_len_means.append(pre_len_mean)
    pre_len_stds.append(pre_len_std)
    pre_len_maxs.append(pre_len_max)
    pre_len_mins.append(pre_len_min)
    title_is_in_pres.append(title_is_in_pre)
    title_in_pre_scores.append(title_in_pre_score)
    in_query_bigs.append(in_query_big)
    title_in_pre_score_sub_maxs.append(title_in_pre_score_sub_max)
    title_in_pre_score_sub_mins.append(title_in_pre_score_sub_min)
    title_in_pre_score_sub_means.append(title_in_pre_score_sub_mean)
    title_len_sub_maxs.append(title_len_sub_max)
    title_len_sub_mins.append(title_len_sub_min)
    title_len_sub_means.append(title_len_sub_mean)
    zh_pre_title_common_words_cnt_means.append(zh_pre_title_common_words_cnt_mean)
    zh_pre_title_common_words_cnt_stds.append(zh_pre_title_common_words_cnt_std)
    zh_pre_title_common_words_cnt_maxs.append(zh_pre_title_common_words_cnt_max)
    zh_pre_title_common_words_cnt_mins.append(zh_pre_title_common_words_cnt_min)
    pre_title_common_words_cnt_means.append(pre_title_common_words_cnt_mean)
    pre_title_common_words_cnt_stds.append(pre_title_common_words_cnt_std)
    pre_title_common_words_cnt_maxs.append(pre_title_common_words_cnt_max)
    pre_title_common_words_cnt_mins.append(pre_title_common_words_cnt_min)
    pre_title_common_words_cnt_mean_rates.append(pre_title_common_words_cnt_mean_rate)
    pre_title_common_words_cnt_max_rates.append(pre_title_common_words_cnt_max_rate)
    pre_title_common_words_cnt_min_rates.append(pre_title_common_words_cnt_min_rate)
    max_pre_title_common_words_cnts.append(max_pre_title_common_words_cnt)
    zh_max_pre_title_common_words_cnts.append(zh_max_pre_title_common_words_cnt)
    max_pre_title_common_words_cnt_rates.append(max_pre_title_common_words_cnt_rate)

    pre_len_w_means.append(pre_len_w_mean)
    title_len_sub_w_means.append(title_len_sub_w_mean)
    prefix_is_in_pres.append(prefix_is_in_pre)
    prefix_in_pre_scores.append(prefix_in_pre_score)
    prefix_in_pre_score_sub_maxs.append(prefix_in_pre_score_sub_max)
    prefix_in_pre_score_sub_mins.append(prefix_in_pre_score_sub_min)
    prefix_in_pre_score_sub_means.append(prefix_in_pre_score_sub_mean)
    prefix_len_sub_maxs.append(prefix_len_sub_max)
    prefix_len_sub_mins.append(prefix_len_sub_min)
    prefix_len_sub_means.append(prefix_len_sub_mean)
    prefix_len_sub_w_means.append(prefix_len_sub_w_mean)
    prefix_len_rate_maxs.append(prefix_len_rate_max)
    prefix_len_rate_mins.append(prefix_len_rate_min)
    prefix_len_rate_means.append(prefix_len_rate_mean)
    prefix_len_rate_w_means.append(prefix_len_rate_w_mean)
    max_pre_prefix_common_words_cnt_rates.append(max_pre_prefix_common_words_cnt_rate)
    zh_max_pre_prefix_common_words_cnt_rates.append(zh_max_pre_prefix_common_words_cnt_rate)

    pre_len_new_w_means.append(pre_len_new_w_mean)
    title_len_sub_new_w_means.append(title_len_sub_new_w_mean)

    # 添加：概率的比值
    title_in_pre_score_max_rates.append(title_in_pre_score_max_rate)
    title_in_pre_score_min_rates.append(title_in_pre_score_min_rate)
    title_in_pre_score_mean_rates.append(title_in_pre_score_mean_rate)

    # 添加：长度的比值
    title_len_max_rates.append(title_len_max_rate)
    title_len_min_rates.append(title_len_min_rate)
    title_len_mean_rates.append(title_len_mean_rate)
    title_len_w_mean_rates.append(title_len_w_mean_rate)
    title_len_new_w_mean_rates.append(title_len_new_w_mean_rate)

    # 添加权重因素的共同字符
    pre_title_common_words_cnt_w_means.append(pre_title_common_words_cnt_w_mean)

    # 添加权重因素的共同字符的比值
    pre_title_common_words_cnt_w_mean_rates.append(pre_title_common_words_cnt_w_mean_rate)

    ##common word rate in prediction##
    pre_title_common_words_rate_means.append(pre_title_common_words_rate_mean)
    pre_title_common_words_rate_maxs.append(pre_title_common_words_rate_max)
    pre_title_common_words_rate_mins.append(pre_title_common_words_rate_min)
    pre_title_common_words_rate_stds.append(pre_title_common_words_rate_std)
    pre_title_common_words_rate_w_means.append(pre_title_common_words_rate_w_mean)

    zh_pre_title_common_words_rate_means.append(zh_pre_title_common_words_rate_mean)
    zh_pre_title_common_words_rate_maxs.append(zh_pre_title_common_words_rate_max)
    zh_pre_title_common_words_rate_mins.append(zh_pre_title_common_words_rate_min)
    zh_pre_title_common_words_rate_stds.append(zh_pre_title_common_words_rate_std)
    # zh_pre_title_common_words_rate_w_means.append(zh_pre_title_common_words_rate_w_mean)

    max_pre_title_common_words_rates.append(max_pre_title_common_words_rate)

    query_t_recall_chars.append(query_t_recall_char)
    query_t_precision_chars.append(query_t_precision_char)
    query_t_acc_chars.append(query_t_acc_char)
    query_t_recall_words.append(query_t_recall_word)
    query_t_precision_words.append(query_t_precision_word)
    query_t_acc_words.append(query_t_acc_word)
    query_t_recall_char_ngrams.append(query_t_recall_char_ngram)
    query_t_precision_char_ngrams.append(query_t_precision_char_ngram)
    query_t_acc_char_ngrams.append(query_t_acc_char_ngram)
    query_t_recall_word_ngrams.append(query_t_recall_word_ngram)
    query_t_precision_word_ngrams.append(query_t_precision_word_ngram)
    query_t_acc_word_ngrams.append(query_t_acc_word_ngram)

print("0%")
query_prediction_feat['pre_mean'] = pre_means
query_prediction_feat['pre_std'] = pre_stds
query_prediction_feat['pre_max'] = pre_maxs
query_prediction_feat['pre_min'] = pre_mins
query_prediction_feat['pre_cnt'] = pre_cnts
print("10%")
query_prediction_feat['pre_len_mean'] = pre_len_means
query_prediction_feat['pre_len_std'] = pre_len_stds
query_prediction_feat['pre_len_max'] = pre_len_maxs
query_prediction_feat['pre_len_min'] = pre_len_mins
print("20%")
query_prediction_feat['title_is_in_pre'] = title_is_in_pres
query_prediction_feat['title_in_pre_score'] = title_in_pre_scores
query_prediction_feat['in_query_big'] = in_query_bigs
query_prediction_feat['title_in_pre_score_sub_max'] = title_in_pre_score_sub_maxs
query_prediction_feat['title_in_pre_score_sub_min'] = title_in_pre_score_sub_mins
query_prediction_feat['title_in_pre_score_sub_mean'] = title_in_pre_score_sub_means
query_prediction_feat['title_len_sub_max'] = title_len_sub_maxs
query_prediction_feat['title_len_sub_min'] = title_len_sub_mins
query_prediction_feat['title_len_sub_mean'] = title_len_sub_means
print("30%")
query_prediction_feat['zh_pre_title_common_words_cnt_mean'] = zh_pre_title_common_words_cnt_means
query_prediction_feat['zh_pre_title_common_words_cnt_std'] = zh_pre_title_common_words_cnt_stds
query_prediction_feat['zh_pre_title_common_words_cnt_max'] = zh_pre_title_common_words_cnt_maxs
query_prediction_feat['zh_pre_title_common_words_cnt_min'] = zh_pre_title_common_words_cnt_mins
query_prediction_feat['pre_title_common_words_cnt_mean'] = pre_title_common_words_cnt_means
query_prediction_feat['pre_title_common_words_cnt_std'] = pre_title_common_words_cnt_stds
query_prediction_feat['pre_title_common_words_cnt_max'] = pre_title_common_words_cnt_maxs
query_prediction_feat['pre_title_common_words_cnt_min'] = pre_title_common_words_cnt_mins
print("40%")
query_prediction_feat['pre_title_common_words_cnt_mean_rate'] = pre_title_common_words_cnt_mean_rates
query_prediction_feat['pre_title_common_words_cnt_max_rate'] = pre_title_common_words_cnt_max_rates
query_prediction_feat['pre_title_common_words_cnt_min_rate'] = pre_title_common_words_cnt_min_rates
query_prediction_feat['max_pre_title_common_words_cnt'] = max_pre_title_common_words_cnts
query_prediction_feat['zh_max_pre_title_common_words_cnt'] = zh_max_pre_title_common_words_cnts
query_prediction_feat['max_pre_title_common_words_cnt_rate'] = max_pre_title_common_words_cnt_rates

print("50%")
query_prediction_feat['pre_len_w_mean'] = pre_len_w_means
query_prediction_feat['title_len_sub_w_mean'] = title_len_sub_w_means
query_prediction_feat['prefix_is_in_pre'] = prefix_is_in_pres
query_prediction_feat['prefix_in_pre_score'] = prefix_in_pre_scores
query_prediction_feat['prefix_in_pre_score_sub_max'] = prefix_in_pre_score_sub_maxs
query_prediction_feat['prefix_in_pre_score_sub_min'] = prefix_in_pre_score_sub_mins
query_prediction_feat['prefix_in_pre_score_sub_mean'] = prefix_in_pre_score_sub_means
print("60%")
query_prediction_feat['prefix_len_sub_max'] = prefix_len_sub_maxs
query_prediction_feat['prefix_len_sub_min'] = prefix_len_sub_mins
query_prediction_feat['prefix_len_sub_mean'] = prefix_len_sub_means
query_prediction_feat['prefix_len_sub_w_mean'] = prefix_len_sub_w_means
query_prediction_feat['prefix_len_rate_max'] = prefix_len_rate_maxs
query_prediction_feat['prefix_len_rate_min'] = prefix_len_rate_mins
query_prediction_feat['prefix_len_rate_mean'] = prefix_len_rate_means
query_prediction_feat['prefix_len_rate_w_mean'] = prefix_len_rate_w_means
print("70%")
query_prediction_feat['max_pre_prefix_common_words_cnt_rate'] = max_pre_prefix_common_words_cnt_rates
query_prediction_feat['zh_max_pre_prefix_common_words_cnt_rate'] = zh_max_pre_prefix_common_words_cnt_rates

query_prediction_feat['pre_len_new_w_mean'] = pre_len_new_w_means
query_prediction_feat['title_len_sub_new_w_mean'] = title_len_sub_new_w_means

# 添加：概率的比值
query_prediction_feat['title_in_pre_score_max_rate'] = title_in_pre_score_max_rates
query_prediction_feat['title_in_pre_score_min_rate'] = title_in_pre_score_min_rates
query_prediction_feat['title_in_pre_score_mean_rate'] = title_in_pre_score_mean_rates

#添加：长度的比值
query_prediction_feat['title_len_max_rate'] = title_len_max_rates
query_prediction_feat['title_len_min_rate'] = title_len_min_rates
query_prediction_feat['title_len_mean_rate'] = title_len_mean_rates
query_prediction_feat['title_len_w_mean_rate'] = title_len_w_mean_rates
query_prediction_feat['title_len_new_w_mean_rate'] = title_len_new_w_mean_rates

#添加权重因素的共同字符
query_prediction_feat['pre_title_common_words_cnt_w_mean'] = pre_title_common_words_cnt_w_means

#添加权重因素的共同字符的比值
query_prediction_feat['pre_title_common_words_cnt_w_mean_rate'] = pre_title_common_words_cnt_w_mean_rates

print("90%")
##common word rate in prediction##
query_prediction_feat['pre_title_common_words_rate_mean'] = pre_title_common_words_rate_means
query_prediction_feat['pre_title_common_words_rate_max'] = pre_title_common_words_rate_maxs
query_prediction_feat['pre_title_common_words_rate_min'] = pre_title_common_words_rate_mins
query_prediction_feat['pre_title_common_words_rate_std'] = pre_title_common_words_rate_stds
query_prediction_feat['pre_title_common_words_rate_w_mean'] = pre_title_common_words_rate_w_means

query_prediction_feat['zh_pre_title_common_words_rate_mean'] = zh_pre_title_common_words_rate_means
query_prediction_feat['zh_pre_title_common_words_rate_max'] = zh_pre_title_common_words_rate_maxs
query_prediction_feat['zh_pre_title_common_words_rate_min'] = zh_pre_title_common_words_rate_mins
query_prediction_feat['zh_pre_title_common_words_rate_std'] = zh_pre_title_common_words_rate_stds
# query_prediction_feat['zh_pre_title_common_words_rate_w_mean'] = zh_pre_title_common_words_rate_w_means

query_prediction_feat['max_pre_title_common_words_rate'] = max_pre_title_common_words_rates

query_prediction_feat['query_t_recall_char'] = query_t_recall_chars
query_prediction_feat['query_t_precision_char'] = query_t_precision_chars
query_prediction_feat['query_t_acc_char'] = query_t_acc_chars
query_prediction_feat['query_t_recall_word'] = query_t_recall_words
query_prediction_feat['query_t_precision_word'] = query_t_precision_words
query_prediction_feat['query_t_acc_word'] = query_t_acc_words
query_prediction_feat['query_t_recall_char_ngram'] = query_t_recall_char_ngrams
query_prediction_feat['query_t_precision_char_ngram'] = query_t_precision_char_ngrams
query_prediction_feat['query_t_acc_char_ngram'] = query_t_acc_char_ngrams
query_prediction_feat['query_t_recall_word_ngram'] = query_t_recall_word_ngrams
query_prediction_feat['query_t_precision_word_ngram'] = query_t_precision_word_ngrams
query_prediction_feat['query_t_acc_word_ngram'] = query_t_acc_word_ngrams

print (query_prediction_feat.shape)

query_prediction_feat.to_csv(query_prediction_feature_txt, sep='\t', index=False)


