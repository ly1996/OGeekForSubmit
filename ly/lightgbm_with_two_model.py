import pandas as pd
import numpy as np
from sklearn import preprocessing
import lightgbm as lgb
import gc
import re
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, f1_score
from time import ctime
import os, sys

# 数据路径
import path_file


######################################################
class __redirection__:
    def __init__(self):
        self.buff = ''
        self.__console__ = sys.stdout

    def write(self, output_stream):
        self.buff += output_stream

    def to_console(self):
        sys.stdout = self.__console__
        print (self.buff)
        sys.stdout = self

    def to_file(self, file_path):
        f = open(file_path, 'a+')
        sys.stdout = f
        print (self.buff)
        f.close()
        sys.stdout = self

    def flush(self):
        self.buff = ''

    def reset(self):
        sys.stdout = self.__console__


######################
# load data
######################
train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
query_prediction_feature_txt = path_file.query_prediction_feature_txt
title_prediction_jaccard_distance_txt = path_file.title_prediction_jaccard_distance_new_txt
title_prediction_distance_txt = path_file.title_prediction_distance_new_txt
title_tag_word2vec_distance_txt = path_file.title_tag_word2vec_distance_new_txt
model_path = path_file.model_path
nlp_feat_txt = path_file.nlp_feat_new_txt

now = ctime()
log_txt = 'out' + now + '.log'
r_obj = __redirection__()
sys.stdout = r_obj

train_data = pd.read_table(train_txt, names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None,
                           encoding='utf-8', quoting=3).astype(str)
val_data = pd.read_table(val_txt,
                         names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                         quoting=3).astype(str)
test_data = pd.read_table(test_txt,
                          names=['prefix', 'query_prediction', 'title', 'tag'], header=None, encoding='utf-8',
                          quoting=3).astype(str)
# 去噪，只有一条数据
train_data = train_data[train_data['label'].isin(['0', '1'])]

prefixs = pd.concat([train_data, val_data]).groupby(['prefix'], as_index=False)['prefix'].agg({'cnt': 'count'})[
    'prefix'].tolist()
print('prefixs : ', len(prefixs))

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

#####################################################
# basic feat
#####################################################

###title feat##########
data['title_len'] = data['title'].apply(lambda x: len(str(x)))
data['title_digit_len'] = data['title'].apply(lambda x: sum(c.isdigit() for c in x))
data['title_alpha_len'] = data['title'].apply(lambda x: len(re.findall('[a-zA-Z]', x)))
data['title_zh_len'] = data['title'].apply(lambda x: len(re.findall(u'([\u4e00-\u9fff])', x)))

data['title_digit_rate'] = data['title_digit_len'] * 1.0 / data['title_len'] * 1.0
data['title_digit_sub'] = data['title_len'] - data['title_digit_len']
data['title_alpha_rate'] = data['title_alpha_len'] * 1.0 / data['title_len'] * 1.0
data['title_alpha_sub'] = data['title_len'] - data['title_alpha_len']

# query_prediction是否为空
data['perdiction_is_null'] = data['query_prediction'].apply(lambda x: 1 if x == 'nan' else 0)

####prefix feat#########
data['prefix_len'] = data['prefix'].apply(lambda x: len(str(x)))
data['prefix_digit_len'] = data['prefix'].apply(lambda x: sum(c.isdigit() for c in x))
data['prefix_alpha_len'] = data['prefix'].apply(lambda x: len(re.findall('[a-zA-Z]', x)))
data['prefix_zh_len'] = data['prefix'].apply(lambda x: len(re.findall(u'([\u4e00-\u9fff])', x)))

data['prefix_digit_rate'] = data['prefix_digit_len'] * 1.0 / data['prefix_len'] * 1.0
data['prefix_digit_sub'] = data['prefix_len'] - data['prefix_digit_len']
data['prefix_alpha_rate'] = data['prefix_alpha_len'] * 1.0 / data['prefix_len'] * 1.0
data['prefix_alpha_sub'] = data['prefix_len'] - data['prefix_alpha_len']

######title_prefix feat##########
data['prefix_is_title'] = data.apply(lambda x: 1 if x['prefix'] == x['title'] else 0, axis=1)
data['common_words_cnt'] = data.apply(
    lambda x: len(set(list(str(x['prefix']))).intersection(set(list(str(x['title']))))), axis=1)
data['common_words_cnt_rate'] = data['common_words_cnt'] / data['title_len']
data['prefix_len_rate'] = data['prefix_len'] / data['title_len']
data['prefix_len_sub'] = data['prefix_len'] - data['title_len']
data['prefix_is_titles_prefix'] = data.apply(lambda x: 1 if str(x['title']).startswith(str(x['prefix'])) else 0, axis=1)

######query predict feat#######

##需要先跑其他两个.py文件（gen_query_prediction_feat.py 和 gen_query_prediction_feat2.py），才会生成以下两个.txt文件
query_prediction_feature = pd.read_csv(query_prediction_feature_txt, sep='\t')
data = pd.concat([data, query_prediction_feature], axis=1)
del query_prediction_feature
gc.collect()

#######word2vec distance#######
####nlp 相关特征，暂时没有怎么做，以上都是统计特征
####这部分特征，就是计算title prefix predict_query的word embedding的距离，
####分词用的是thulac，word2vec用的是前几天腾讯开源的中文embedding，地址https://ai.tencent.com/ailab/nlp/embedding.html
####在这个baseline的性能上没有提升多少，不到1个点，可以暂时忽略，跑之前也需要生成以下三个txt文件##

nlp_feat = pd.read_csv(nlp_feat_txt, sep='\t')
data = pd.concat([data, nlp_feat], axis=1)
del nlp_feat
gc.collect()

title_prediction_jaccard_distance = pd.read_csv(title_prediction_jaccard_distance_txt, sep='\t')
data = pd.concat([data, title_prediction_jaccard_distance], axis=1)
del title_prediction_jaccard_distance
gc.collect()

title_prediction_distance = pd.read_csv(title_prediction_distance_txt, sep='\t')
data = pd.concat([data, title_prediction_distance], axis=1)
del title_prediction_distance
gc.collect()

title_tag_word2vec_distance = pd.read_csv(title_tag_word2vec_distance_txt, sep='\t')
data = pd.concat([data, title_tag_word2vec_distance], axis=1)
del title_tag_word2vec_distance
gc.collect()

#######CTR feat##########

#####感觉 CTR特征很容易过拟合，尤其是数据量少的时候，所以再跑的时候没有用######

# 打乱数据并分成num_folds份
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

train_val_data = data[:-len_test]
train_val_data = shuffle(train_val_data, random_state=103)

len_train_val = train_val_data.shape[0]
num_folds = 5
step = int(len_train_val / num_folds) + 1
data_slices = []

from sklearn.preprocessing import MinMaxScaler

items = ['prefix', 'title', 'tag']
# def get_ctr_feat(feat_data,tmp_data):
#     for item in items:
#         temp = feat_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
#         temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
#         tmp_data = pd.merge(tmp_data, temp, on=item, how='left')
#         tmp_data = tmp_data.fillna(0)
#         mm = MinMaxScaler()
#         tmp_data[[item + '_click',item + '_count']] = mm.fit_transform(tmp_data[[item + '_click',item + '_count']])
#
#     print('2 cross')
#     for i in range(len(items)):
#         for j in range(i + 1, len(items)):
#             item_g = [items[i], items[j]]
#             temp = feat_data.groupby(item_g, as_index=False)['label'].agg(
#                 {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
#             temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (
#                 temp['_'.join(item_g) + '_count'] + 3)
#             tmp_data = pd.merge(tmp_data, temp, on=item, how='left')
#
#             tmp_data = tmp_data.fillna(0)
#             mm = MinMaxScaler()
#             tmp_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
#                 tmp_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])
#     return tmp_data
#
# folds = KFold(len_train_val,n_folds = num_folds,shuffle = True,random_state=42)
# for curr_fold, (idx_train,idx_val) in enumerate(folds):
#     print ("curr fold : ",curr_fold)
#     feat_data = train_val_data.loc[idx_train].copy()
#     tmp_data = train_val_data.loc[idx_val].copy()
#     data_slices.append(get_ctr_feat(feat_data,tmp_data))

for i in range(0, len_train_val, step):
    ith_data = train_val_data[i:i + step]
    print (ith_data.shape[0])
    rest_data = pd.concat([train_val_data[:i], train_val_data[i + step:]])

    for item in items:
        temp = rest_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
        temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
        ith_data = pd.merge(ith_data, temp, on=item, how='left')
        ith_data = ith_data.fillna(0)
        mm = MinMaxScaler()
        ith_data[[item + '_click', item + '_count']] = mm.fit_transform(ith_data[[item + '_click', item + '_count']])
    print('2 cross')
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_g = [items[i], items[j]]
            temp = rest_data.groupby(item_g, as_index=False)['label'].agg(
                {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
            temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (
                temp['_'.join(item_g) + '_count'] + 3)
            ith_data = pd.merge(ith_data, temp, on=item_g, how='left')
            ith_data = ith_data.fillna(0)
            mm = MinMaxScaler()
            ith_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
                ith_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])

    print('3 cross')
    item_g = ['prefix', 'title', 'tag']
    temp = rest_data.groupby(item_g, as_index=False)['label'].agg(
        {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
    temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (
        temp['_'.join(item_g) + '_count'] + 3)
    ith_data = pd.merge(ith_data, temp, on=item_g, how='left')
    ith_data = ith_data.fillna(0)
    mm = MinMaxScaler()
    ith_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
        ith_data[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])
    data_slices.append(ith_data)

train_val_data_ctr = pd.concat(data_slices, ignore_index=True)
del data_slices
gc.collect()

train_val_data = pd.concat([train_data, val_data], ignore_index=True)
train_val_data['label'] = train_val_data['label'].apply(lambda x: int(x))

test_data_ctr = data[-len_test:]
# test_data_ctr = get_ctr_feat(train_val_data,test_data_ctr)

for item in items:
    print(item)
    temp = train_val_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
    temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
    temp[item + '_click'] = temp[item + '_click'] / num_folds * (num_folds - 1)
    temp[item + '_count'] = temp[item + '_count'] / num_folds * (num_folds - 1)
    test_data_ctr = pd.merge(test_data_ctr, temp, on=item, how='left')
    test_data_ctr = test_data_ctr.fillna(0)
    mm = MinMaxScaler()
    test_data_ctr[[item + '_click', item + '_count']] = mm.fit_transform(
        test_data_ctr[[item + '_click', item + '_count']])
print('2 cross')
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        print(items[i], ' ', items[j])
        item_g = [items[i], items[j]]
        temp = train_val_data.groupby(item_g, as_index=False)['label'].agg(
            {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
        temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (temp['_'.join(item_g) + '_count'] + 3)
        temp['_'.join(item_g) + '_click'] = temp['_'.join(item_g) + '_click'] / num_folds * (num_folds - 1)
        temp['_'.join(item_g) + '_count'] = temp['_'.join(item_g) + '_count'] / num_folds * (num_folds - 1)
        test_data_ctr = pd.merge(test_data_ctr, temp, on=item_g, how='left')

        test_data_ctr = test_data_ctr.fillna(0)
        mm = MinMaxScaler()
        test_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
            test_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])

print('3 cross')
item_g = ['prefix', 'title', 'tag']
temp = train_val_data.groupby(item_g, as_index=False)['label'].agg(
    {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (temp['_'.join(item_g) + '_count'] + 3)
temp['_'.join(item_g) + '_click'] = temp['_'.join(item_g) + '_click'] / num_folds * (num_folds - 1)
temp['_'.join(item_g) + '_count'] = temp['_'.join(item_g) + '_count'] / num_folds * (num_folds - 1)
test_data_ctr = pd.merge(test_data_ctr, temp, on=item_g, how='left')
test_data_ctr = test_data_ctr.fillna(0)
mm = MinMaxScaler()
test_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
    test_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])

data = pd.concat([train_val_data_ctr, test_data_ctr])
del train_data
del val_data
del test_data
del train_val_data
del train_val_data_ctr
del test_data_ctr
del ith_data
del temp
del rest_data
gc.collect()


#################
# encode 将各种标签分配一个可数的连续编号，tag最需要
##################
def encode_count_with_train(train, df, column_name, new_column_name=''):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[column_name].values))
    try:
        df[new_column_name] = lbl.transform(list(df[column_name].values))
    except ValueError:
        print("LabelEncoder_list transform out range.")
        x_test = []
        test_values = df[column_name].values
        feat_len = len(test_values)
        fit_len = len(lbl.classes_)

        for j in range(feat_len):
            if test_values[j] in lbl.classes_:
                # 看当前value与fit的数据集是否有交集
                x_test.append(np.searchsorted(lbl.classes_, test_values[j]))
                # 如果有，把fit的编号返回
            else:
                x_test.append(fit_len + 2)
                # 没有则返回一个fit中没有的编号
                # 编号为 0 - (fit_len - 1)
        x_test = np.array(x_test)
        df[new_column_name] = x_test
    return df


def encode_count(df, column_name, new_column_name=''):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    if new_column_name == '':
        df[column_name] = lbl.transform(list(df[column_name].values))
    else:
        df[new_column_name] = lbl.transform(list(df[column_name].values))
    return df


encoder_col = ['prefix', 'title', 'tag']
for col in encoder_col:
    print(col)
    data = encode_count(data, col, col + '_id')
# print('Start encoder_col ： %s' % (ctime()))
# for col in encoder_col:
#     print(col)
#     train_val_data_ctr = encode_count_with_train(train_val_data_ctr,train_val_data_ctr,col,col+'_id')
#     test_data_ctr = encode_count_with_train(train_val_data_ctr, test_data_ctr, col, col + '_id')
#
# print('End encoder_col ： %s' % (ctime()))
#
# data = pd.concat([train_val_data_ctr,test_data_ctr])

r_obj.to_console()
r_obj.flush()

#######################
##drop out useless feat
#######################
drop_feat = [
    'prefix',
    'query_prediction',
    'title',
    'tag',
    'label',
    'prefix_click',
    #  'prefix_count',
    #  'prefix_ctr',
    'title_click',
    #  'title_count',
    #  'title_ctr',
    'tag_click',
    #  'tag_count',
    #  'tag_ctr',
    'query_prediction_click',
    #  'query_prediction_count',
    'query_prediction_ctr',
    'prefix_title_click',
    #  'prefix_title_count',
    #  'prefix_title_ctr',
    'prefix_tag_click',
    #  'prefix_tag_count',
    #  'prefix_tag_ctr',
    'prefix_query_prediction_click',
    #  'prefix_query_prediction_count',
    'prefix_query_prediction_ctr',
    'title_tag_click',
    #  'title_tag_count',
    #  'title_tag_ctr',
    'title_query_prediction_click',
    #  'title_query_prediction_count',
    'title_query_prediction_ctr',
    'tag_query_prediction_click',
    #  'tag_query_prediction_count',
    'tag_query_prediction_ctr',
    'prefix_title_tag_click',
    #  'prefix_title_tag_count',
    #  'prefix_title_tag_ctr',
    # 'prefix_len_mul_pre_std',
    'title_prefix_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_prewmean_jaccard_distance'
]

drop_feat2 = [
    'prefix',
    'query_prediction',
    'title',
    'tag',
    'label',
    'prefix_click',
    'prefix_count',
    'prefix_ctr',
    'title_click',
    'title_count',
    'title_ctr',
    'tag_click',
    #  'tag_count',
    #  'tag_ctr',
    'query_prediction_click',
    'query_prediction_count',
    'query_prediction_ctr',
    'prefix_title_click',
    'prefix_title_count',
    'prefix_title_ctr',
    'prefix_tag_click',
    'prefix_tag_count',
    'prefix_tag_ctr',
    'prefix_query_prediction_click',
    'prefix_query_prediction_count',
    'prefix_query_prediction_ctr',
    'title_tag_click',
    'title_tag_count',
    'title_tag_ctr',
    'title_query_prediction_click',
    'title_query_prediction_count',
    'title_query_prediction_ctr',
    'tag_query_prediction_click',
    'tag_query_prediction_count',
    'tag_query_prediction_ctr',
    'prefix_title_tag_click',
    'prefix_title_tag_count',
    'prefix_title_tag_ctr',
    # 'prefix_len_mul_pre_std',s
    'title_prefix_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_premean_jaccard_distance',
    'title_prewmean_jaccard_distance',
    'title_prewmean_jaccard_distance'
]

###################################
# train
###################################

nfold = 5

# 类别特征，不需要转化为one-hot，指明后lightgbm可以直接处理
categorical_columns_can = []

drop_cols_can = drop_feat
drop_cols = []
data_feat = list(data.columns)
categorical_columns = []
for i in categorical_columns_can:
    if i in data_feat and i not in drop_cols:
        categorical_columns.append(i)

for i in drop_cols_can:
    if i in data_feat:
        drop_cols.append(i)

drop_cols_can2 = drop_feat2
drop_cols2 = []

for i in drop_cols_can2:
    if i in data_feat:
        drop_cols2.append(i)

num_round = 100000

# 阈值
online_threshold = 0.38

# 可以测试不同的learning rate
# 0.005,0.01,0.05
for lr in [0.05]:
    # 可以测试不同的rand seed
    for rand_seed in [100]:
        print('rand_seed ', rand_seed)

        train_feat = data[:-len_test].copy()
        test_feat = data[-len_test:].copy()

        del data
        gc.collect()

        print('train shape:', train_feat.shape)
        print('test shape:', test_feat.shape)

        # train model 1
        data_x = train_feat.drop(drop_cols, axis=1).values
        data_y = train_feat['label'].values

        # train model 2
        data_x2 = train_feat.drop(drop_cols2, axis=1).values
        data_y2 = data_y

        # data_x_with_pre = train_feat.values

        # real_val_x = val_feat.drop(drop_cols, axis=1).values
        # real_val_y = val_feat['label'].values

        # test_x = test_feat.drop(drop_cols, axis=1).values
        test_x_in = test_feat.drop(drop_cols, axis=1).values
        test_x_not_in = test_feat.drop(drop_cols2, axis=1).values

        print ("test is in prefix : ", test_feat[test_feat['prefix'].isin(prefixs)].shape)

        SEED = rand_seed

        # feat_colums：model 1输入的特征
        feat_colums = list(train_feat.drop(drop_cols, axis=1).columns)
        print(feat_colums)
        print(len(feat_colums))

        # feat_colums：model 2输入的特征
        feat_colums2 = list(train_feat.drop(drop_cols2, axis=1).columns)
        print(feat_colums2)
        print(len(feat_colums2))

        all_colums = list(train_feat.columns)

        del train_feat
        gc.collect()

        folds = KFold(n_splits=nfold, shuffle=True, random_state=42)

        global_best_score = 0
        global_best_f1_score = 0

        global_best_score2 = 0
        global_best_f1_score2 = 0

        pred_results = []
        pred_results_in = []
        pred_results_not_in = []

        r_obj.to_console()
        r_obj.to_file(log_txt)
        r_obj.flush()
        #
        # train_pre = np.zeros((len(data_x)))
        # cv_dd = []

        for curr_fold, (idx_train, idx_val) in enumerate(folds.split(data_x)):
            print('cur_fold:', curr_fold)
            print('lr :', lr)
            print('seed :', rand_seed)

            # 当前训练集包含的prefix set
            # prefixs = pd.DataFrame(data_x_with_pre[idx_train], columns=all_colums).groupby(['prefix'], as_index=False)['prefix'].agg({'cnt': 'count'})[
            #     'prefix'].tolist()
            # print('prefixs : ', len(prefixs))

            params = {
                # num_leaves：128，256
                'num_leaves': 2 ** 7 - 1,
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'min_data_in_leaf': 50,
                'learning_rate': lr,
                'feature_fraction': 0.65,
                'bagging_fraction': 0.75,
                'bagging_freq': 1,
                'metric': {'binary_logloss'},
                'seed': SEED,
                # 'scale_pos_weight':0.899844466771833,
                # 'min_child_weight':5,
                # 'min_split_gain':0,
                # 'subsample_for_bin':50000,
                'nthread': 15,
                #                 'lambda_l1':3,
                #                 'lambda_l2':2,
                'max_bin': 1023,
                # 'device': 'gpu'
            }
            print(params)

            train_x = data_x[idx_train]
            train_y = data_y[idx_train]

            val_x = data_x[idx_val]
            val_y = data_y[idx_val]

            print('feat num:', train_x.shape)

            train_matrix = lgb.Dataset(pd.DataFrame(train_x, columns=feat_colums), label=train_y,
                                       categorical_feature=categorical_columns)
            valid_matrix = lgb.Dataset(pd.DataFrame(val_x, columns=feat_colums), label=val_y,
                                       categorical_feature=categorical_columns)

            early_stopping_rounds = 100
            model = lgb.train(params, train_matrix, num_round, valid_sets=[valid_matrix],
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=500
                              )
            best_iter = model.best_iteration

            # os.path.expanduser('~/Disk/OGeekModel/model1/model'+str(curr_fold)+'.txt')
            # model.save_model('data/model2/model_with_ctr' + str(curr_fold) + '.txt', model.best_iteration)
            model.save_model(os.path.expanduser(model_path + 'model_with_ctr' + str(curr_fold) + '.txt'),
                             model.best_iteration)

            print('val_pred')
            # 利用模型对分出来的一折验证集做预测,得到0到1之间的小数
            val_pred = model.predict(val_x, num_iteration=best_iter)
            train_pred = model.predict(train_x, num_iteration=best_iter)

            local_best_score = log_loss(val_y, val_pred)
            local_best_f1_score = f1_score(val_y, np.where(val_pred > online_threshold, 1, 0))
            global_best_score += local_best_score
            global_best_f1_score += local_best_f1_score

            print('test_pred')
            # 利用模型对无标测试集做预测
            test_pred_in = model.predict(test_x_in, num_iteration=best_iter)

            print('mean pre score:', np.mean(test_pred_in))
            print('local_best_score:', local_best_score)
            print('global_best_score:', global_best_score / (curr_fold + 1))
            print('local_best_f1_score:', local_best_f1_score)
            print('global_best_f1_score:', global_best_f1_score / (curr_fold + 1))
            # 对模型在一折验证集上的分类情况做分析
            print(classification_report(val_y, np.where(val_pred > online_threshold, 1, 0), digits=6))
            print(classification_report(train_y, np.where(train_pred > online_threshold, 1, 0), digits=6))

            # 测试不同的阈值
            max_f1 = 0
            max_threshold = 0
            for threshold in np.arange(0.34, 0.66, 0.01):
                tmp_f1 = f1_score(val_y, np.where(val_pred > threshold, 1, 0))
                print(threshold, ' f1 score: ', tmp_f1)
                if tmp_f1 > max_f1:
                    max_f1 = tmp_f1
                    max_threshold = threshold
            print('best threshold: ', max_threshold, '  f1 :', max_f1)

            # 对feature的重要性排序
            #             feature_importance_spilt = pd.DataFrame(
            #                     {'name': model.feature_name(),
            #                      'importance': model.feature_importance(importance_type='split')}).sort_values(
            #                     by='importance', ascending=False)
            #             feature_importance_gain = pd.DataFrame(
            #                     {'name': model.feature_name(),
            #                      'importance': model.feature_importance(importance_type='gain')}).sort_values(
            #                     by='importance', ascending=False)
            #             dd = feature_importance_gain.merge(feature_importance_spilt, on='name').copy()
            #             dd['t'] = dd['importance_x'] / dd['importance_y']
            #             dd['mul'] = dd['importance_x'] * dd['importance_y']

            # if curr_fold == 0:
            #     pre_val_y = np.where(val_pred > online_threshold, 1, 0)
            #     may_wrong_data = pd.DataFrame(data_x_with_pre[idx_val], columns=all_colums)
            #     may_wrong_data['val_pred'] = val_pred
            #     may_wrong_data['label'] = val_y
            #     may_wrong_data['label_pred'] = pre_val_y
            #     wrong_data = may_wrong_data[may_wrong_data['label_pred'] != may_wrong_data['label']]
            #
            #     wrong_data.to_csv('results/wrong_data_'+str(lr)+'.csv',
            #                       sep=',', index=False)
            #
            # dd.to_csv('results/feature_importance_'+str(lr) + '_' +str(global_best_score/(curr_fold+1))+'_'+str(global_best_f1_score/(curr_fold+1))+'.txt',sep=" ",index=False)

            # cv_dd.append(dd)

            r_obj.to_console()
            r_obj.to_file(log_txt)
            r_obj.flush()

            del model
            del train_x
            del train_y
            del val_x
            del val_y
            del train_matrix
            del valid_matrix
            gc.collect()

            train_x2 = data_x2[idx_train]
            train_y2 = data_y2[idx_train]

            val_x2 = data_x2[idx_val]
            val_y2 = data_y2[idx_val]

            print('feat num 2:', train_x2.shape)

            train_matrix2 = lgb.Dataset(pd.DataFrame(train_x2, columns=feat_colums2), label=train_y2,
                                        categorical_feature=categorical_columns)
            valid_matrix2 = lgb.Dataset(pd.DataFrame(val_x2, columns=feat_colums2), label=val_y2,
                                        categorical_feature=categorical_columns)

            model2 = lgb.train(params, train_matrix2, num_round, valid_sets=[valid_matrix2],
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=500
                               )
            best_iter2 = model2.best_iteration
            # model2.save_model('data/model2/model_without_ctr' + str(curr_fold) + '.txt', model2.best_iteration)
            model2.save_model(model_path + 'model_without_ctr' + str(curr_fold) + '.txt',
                              model2.best_iteration)

            print('val_pred')
            # 利用模型对分出来的一折验证集做预测,得到0到1之间的小数
            val_pred2 = model2.predict(val_x2, num_iteration=best_iter2)
            train_pred2 = model2.predict(train_x2, num_iteration=best_iter2)

            local_best_score2 = log_loss(val_y2, val_pred2)
            local_best_f1_score2 = f1_score(val_y2, np.where(val_pred2 > online_threshold, 1, 0))
            global_best_score2 += local_best_score2
            global_best_f1_score2 += local_best_f1_score2

            print('test_pred')
            # 利用模型对无标测试集做预测
            test_pred_not_in = model2.predict(test_x_not_in, num_iteration=best_iter2)

            print('mean pre score:', np.mean(test_pred_not_in))
            print('local_best_score:', local_best_score2)
            print('global_best_score:', global_best_score2 / (curr_fold + 1))
            print('local_best_f1_score:', local_best_f1_score2)
            print('global_best_f1_score:', global_best_f1_score2 / (curr_fold + 1))
            # 对模型在一折验证集上的分类情况做分析
            print(classification_report(val_y2, np.where(val_pred2 > online_threshold, 1, 0), digits=6))
            print(classification_report(train_y2, np.where(train_pred2 > online_threshold, 1, 0), digits=6))

            # 测试不同的阈值
            max_f1 = 0
            max_threshold = 0
            for threshold in np.arange(0.34, 0.66, 0.01):
                tmp_f1 = f1_score(val_y2, np.where(val_pred2 > threshold, 1, 0))
                print(threshold, ' f1 score: ', tmp_f1)
                if tmp_f1 > max_f1:
                    max_f1 = tmp_f1
                    max_threshold = threshold
            print('best threshold: ', max_threshold, '  f1 :', max_f1)

            # 对feature的重要性排序
            #             feature_importance_spilt = pd.DataFrame(
            #                 {'name': model2.feature_name(),
            #                  'importance': model2.feature_importance(importance_type='split')}).sort_values(
            #                 by='importance', ascending=False)
            #             feature_importance_gain = pd.DataFrame(
            #                 {'name': model2.feature_name(),
            #                  'importance': model2.feature_importance(importance_type='gain')}).sort_values(
            #                 by='importance', ascending=False)
            #             dd = feature_importance_gain.merge(feature_importance_spilt, on='name').copy()
            # dd['t'] = dd['importance_x'] / dd['importance_y']
            # dd['mul'] = dd['importance_x'] * dd['importance_y']
            #
            # if curr_fold == 0:
            #     pre_val_y = np.where(val_pred2 > online_threshold, 1, 0)
            #     may_wrong_data = pd.DataFrame(data_x_with_pre[idx_val], columns=all_colums)
            #     may_wrong_data['val_pred'] = val_pred2
            #     may_wrong_data['label'] = val_y2
            #     may_wrong_data['label_pred'] = pre_val_y
            #     wrong_data = may_wrong_data[may_wrong_data['label_pred'] != may_wrong_data['label']]
            #
            #     wrong_data.to_csv('results2/wrong_data_' + str(lr) + '.csv',
            #                       sep=',', index=False)
            #
            # dd.to_csv('results2/feature_importance_' + str(lr) + '_' + str(
            #     global_best_score / (curr_fold + 1)) + '_' + str(global_best_f1_score / (curr_fold + 1)) + '.txt',
            #           sep=" ", index=False)

            # cv_dd.append(dd)

            r_obj.to_console()
            r_obj.to_file(log_txt)
            r_obj.flush()

            del model2
            del train_x2
            del train_y2
            del val_x2
            del val_y2
            del train_matrix2
            del valid_matrix2
            gc.collect()

            # real_val_y = np.concatenate([real_val_y_in, real_val_y_not_in])
            # real_val_pred_y = np.concatenate([real_val_pred_in, real_val_pred_not_in])
            #
            # print(f1_score(real_val_y, np.where(real_val_pred_y > online_threshold, 1, 0)))
            # print(classification_report(real_val_y, np.where(real_val_pred_y > online_threshold, 1, 0), digits=6))

            final_pred = np.zeros(test_feat.shape[0])
            final_pred[test_feat['prefix'].isin(prefixs)] = test_pred_in[test_feat['prefix'].isin(prefixs)]
            final_pred[~test_feat['prefix'].isin(prefixs)] = test_pred_not_in[~test_feat['prefix'].isin(prefixs)]
            pred_results.append(final_pred)
            pred_results_in.append(test_pred_in)
            pred_results_not_in.append(test_pred_not_in)

        # 对五次模型预测的结果求平均
        res = np.mean(np.vstack(pred_results), axis=0)
        res_in = np.mean(np.vstack(pred_results_in), axis=0)
        res_not_in = np.mean(np.vstack(pred_results_not_in), axis=0)

        label_in = np.where(res_in > 0.38, 1, 0)
        label_not_in = np.where(res_not_in > 0.39, 1, 0)

        label = np.zeros((test_feat.shape[0],), dtype=int)

        label[test_feat['prefix'].isin(prefixs)] = label_in[test_feat['prefix'].isin(prefixs)]
        label[~test_feat['prefix'].isin(prefixs)] = label_not_in[~test_feat['prefix'].isin(prefixs)]

        submission = pd.DataFrame()
        submission['predicted_score'] = res

        submission['label'] = label

        #         submission['label'] = submission['predicted_score'].apply(lambda x: 1 if x > online_threshold else 0)

        submission.to_csv(
            'pred_online_threshold_' + str(online_threshold) + '_feat_' + str(lr) + '_' + str(
                len(feat_colums)) + '_' + str(
                submission['predicted_score'].mean()) + '_' + str(submission['label'].sum()) + '_' + str(
                global_best_score / (curr_fold + 1)) + '_' + str(
                global_best_f1_score / (curr_fold + 1)) + '.csv', sep=',', index=False)
        submission['label'].to_csv(
            'result_online_threshold_' + str(online_threshold) + '_feat_' + str(lr) + '_' + str(
                len(feat_colums)) + '_' + str(
                submission['predicted_score'].mean()) + '_' + str(submission['label'].sum()) + '_' + str(
                global_best_score / (curr_fold + 1)) + '_' + str(
                global_best_f1_score / (curr_fold + 1)) + '.csv', sep=',', index=False)