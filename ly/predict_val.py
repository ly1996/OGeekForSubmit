import pandas as pd
import numpy as np
from sklearn import preprocessing
import lightgbm as lgb
import gc
import re
from sklearn.metrics import classification_report, f1_score

# 数据路径
import path_file

######################################################

######################
# load data
######################
train_txt = path_file.train_txt
test_txt = path_file.test_txt
val_txt = path_file.val_txt
query_prediction_feature_txt = path_file.query_prediction_feature_txt
title_prediction_jaccard_distance_txt = path_file.title_prediction_jaccard_distance_txt
title_prediction_distance_txt = path_file.title_prediction_distance_txt
title_tag_word2vec_distance_txt = path_file.title_tag_word2vec_distance_txt
model_path = '../Models/ly/11.16/'
nlp_feat_txt = path_file.nlp_feat_txt
# model_path = path_file.model_path

train_data = pd.read_table(train_txt,
                           names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                           quoting=3).astype(str)
val_data = pd.read_table(val_txt,
                         names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8',
                         quoting=3).astype(str)
# 去噪，只有一条数据
train_data = train_data[train_data['label'].isin(['0', '1'])]

len_train = train_data.shape[0]
len_val = val_data.shape[0]
len_test = 200000
print("len_train", len_train)
print("len_val", len_val)

# 连接在一起便于统一处理
data = pd.concat([train_data, val_data], ignore_index=True)
data['label'] = data['label'].apply(lambda x: int(x))

prefixs = train_data.groupby(['prefix'], as_index=False)['prefix'].agg({'cnt': 'count'})['prefix'].tolist()
print('prefixs : ', len(prefixs))

# train_data['prefix_title'] = train_data['prefix'].apply(lambda x: x + "_")
# train_data['prefix_title'] = train_data['prefix_title'] + train_data['title']
# prefixs_title = train_data.groupby(['prefix_title'], as_index=False)['prefix_title'].agg({'cnt': 'count'})['prefix_title'].tolist()
# print ("prefixs_title : ",len(prefixs_title))

#####################################################
# basic feat
#####################################################

###title feat##########
print ("gen_basic_feat")
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

##需要先跑其他两个.py文件（gen_query_prediction_feat.py 和 gen_query_prediction_feat2.py），才会生成以下两个.txt文件
query_prediction_feature = pd.read_csv(query_prediction_feature_txt, sep='\t')
data = pd.concat([data, query_prediction_feature[:-len_test]], axis=1)
del query_prediction_feature
gc.collect()

#######word2vec distance#######
####nlp 相关特征，暂时没有怎么做，以上都是统计特征
####这部分特征，就是计算title prefix predict_query的word embedding的距离，
####分词用的是thulac，word2vec用的是前几天腾讯开源的中文embedding，地址https://ai.tencent.com/ailab/nlp/embedding.html
####在这个baseline的性能上没有提升多少，不到1个点，可以暂时忽略，跑之前也需要生成以下三个txt文件##

# nlp_feat = pd.read_csv(nlp_feat_txt,sep='\t')
# data = pd.concat([data, nlp_feat[:-len_test]], axis=1)
# del nlp_feat
# gc.collect()

print ("title_prediction_jaccard_distance")
title_prediction_jaccard_distance = pd.read_csv(title_prediction_jaccard_distance_txt, sep='\t')
data = pd.concat([data, title_prediction_jaccard_distance[:-len_test]], axis=1)
del title_prediction_jaccard_distance
gc.collect()

print ("title_prediction_distance")
title_prediction_distance = pd.read_csv(title_prediction_distance_txt, sep='\t')
data = pd.concat([data, title_prediction_distance[:-len_test]], axis=1)
del title_prediction_distance
gc.collect()

print ("title_tag_word2vec_distance")
title_tag_word2vec_distance = pd.read_csv(title_tag_word2vec_distance_txt, sep='\t')
data = pd.concat([data, title_tag_word2vec_distance[:-len_test]], axis=1)
del title_tag_word2vec_distance
gc.collect()

#######CTR feat##########

#####感觉 CTR特征很容易过拟合，尤其是数据量少的时候，所以再跑的时候没有用######

# 打乱数据并分成num_folds份
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

del train_data
gc.collect()

train_data = data[:-len_val]
train_data = shuffle(train_data)

num_folds = 5
step = int(len_train / num_folds) + 1
data_slices = []

from sklearn.preprocessing import MinMaxScaler

items = ['prefix', 'title', 'tag', 'query_prediction']
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

for i in range(0, len_train, step):
    ith_data = train_data[i:i + step]
    print (ith_data.shape[0])
    rest_data = pd.concat([train_data[:i], train_data[i + step:]])

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

    data_slices.append(ith_data)

train_data_ctr = pd.concat(data_slices, ignore_index=True)
del data_slices
gc.collect()

val_data_ctr = data[-len_val:]

for item in items:
    print(item)
    temp = train_data.groupby(item, as_index=False)['label'].agg({item + '_click': 'sum', item + '_count': 'count'})
    temp[item + '_ctr'] = temp[item + '_click'] / (temp[item + '_count'])
    temp[item + '_click'] = temp[item + '_click'] / num_folds * (num_folds - 1)
    temp[item + '_count'] = temp[item + '_count'] / num_folds * (num_folds - 1)
    val_data_ctr = pd.merge(val_data_ctr, temp, on=item, how='left')
    val_data_ctr = val_data_ctr.fillna(0)
    mm = MinMaxScaler()
    val_data_ctr[[item + '_click', item + '_count']] = mm.fit_transform(
        val_data_ctr[[item + '_click', item + '_count']])
print('2 cross')
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        print(items[i], ' ', items[j])
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg(
            {'_'.join(item_g) + '_click': 'sum', '_'.join(item_g) + '_count': 'count'})
        temp['_'.join(item_g) + '_ctr'] = temp['_'.join(item_g) + '_click'] / (temp['_'.join(item_g) + '_count'] + 3)
        temp['_'.join(item_g) + '_click'] = temp['_'.join(item_g) + '_click'] / num_folds * (num_folds - 1)
        temp['_'.join(item_g) + '_count'] = temp['_'.join(item_g) + '_count'] / num_folds * (num_folds - 1)
        val_data_ctr = pd.merge(val_data_ctr, temp, on=item_g, how='left')

        val_data_ctr = val_data_ctr.fillna(0)
        mm = MinMaxScaler()
        val_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']] = mm.fit_transform(
            val_data_ctr[['_'.join(item_g) + '_click', '_'.join(item_g) + '_count']])

data = pd.concat([train_data_ctr, val_data_ctr])
del train_data
del val_data
del train_data_ctr
del val_data_ctr
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


encoder_col = ['prefix', 'query_prediction', 'title', 'tag']
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
    'tag_count',
    'tag_ctr',
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

# 阈值
online_threshold = 0.38
nfolds = 5
pred_results = []

val_feat = data[-len_val:].copy()

# val_feat['prefix_title'] = val_feat['prefix'].apply(lambda x: x + "_")
# val_feat['prefix_title'] = val_feat['prefix_title'] + val_feat['title']
#
# # val model 1
# val_feat_in = val_feat[val_feat['prefix_title'].isin(prefixs_title)]
# # val model2
# val_feat_not_in = val_feat[~val_feat['prefix_title'].isin(prefixs_title)]
#
# val_feat_in = val_feat_in.drop(['prefix_title'], axis=1)
# val_feat_not_in = val_feat_not_in.drop(['prefix_title'], axis=1)

# val model 1
val_feat_in = val_feat[val_feat['prefix'].isin(prefixs)]
# val model2
val_feat_not_in = val_feat[~val_feat['prefix'].isin(prefixs)]

# 将val数据集的比例调至和test一样
# len_of_not_in = val_feat_not_in.shape[0]
# len_of_in = int(len_of_not_in * 1.4317 + 1)

# val_feat_in = shuffle(val_feat_in,random_state=103)
# val_feat_in = val_feat_in[:len_of_in]

# val model 1
real_val_x_in = val_feat_in.drop(drop_cols, axis=1).values
real_val_y_in = val_feat_in['label'].values

# val model 2
real_val_x_not_in = val_feat_not_in.drop(drop_cols2, axis=1).values
real_val_y_not_in = val_feat_not_in['label'].values

real_val_y = np.concatenate([real_val_y_in, real_val_y_not_in])

print("len of val in prefixs : ", val_feat_in.shape[0])
print("len of val not in prefixs : ", val_feat_not_in.shape[0])
len_of_in = val_feat_in.shape[0]

del data
del val_feat
del val_feat_in
del val_feat_not_in
gc.collect()

for curr_fold in range(nfolds):
    print('cur_fold:', curr_fold)

    model = lgb.Booster(
        model_file=model_path + 'val_model_with_ctr' + str(curr_fold) + '.txt')
    real_val_pred_in = model.predict(real_val_x_in)

    print("different threshold for real val pred in")
    max_f1 = 0
    max_threshold_in = 0
    for threshold in np.arange(0.30, 0.60, 0.01):
        tmp_f1 = f1_score(real_val_y_in, np.where(real_val_pred_in > threshold, 1, 0))
        print(threshold, ' f1 score: ', tmp_f1)
        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_threshold_in = threshold
    print('best threshold: ', max_threshold_in, '  f1 :', max_f1)

    del model
    gc.collect()

    model2 = lgb.Booster(
        model_file=model_path + 'val_model_without_ctr' + str(curr_fold) + '.txt')
    real_val_pred_not_in = model2.predict(real_val_x_not_in)

    print("different threshold for real val pred not in")
    max_f1 = 0
    max_threshold_not_in = 0
    for threshold in np.arange(0.30, 0.60, 0.01):
        tmp_f1 = f1_score(real_val_y_not_in, np.where(real_val_pred_not_in > threshold, 1, 0))
        print(threshold, ' f1 score: ', tmp_f1)
        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_threshold_not_in = threshold
    print('best threshold: ', max_threshold_not_in, '  f1 :', max_f1)

    del model2
    gc.collect()

    real_val_pred_y = np.concatenate([real_val_pred_in, real_val_pred_not_in])
    print ("online_threshold")
    print(f1_score(real_val_y, np.where(real_val_pred_y > online_threshold, 1, 0)))
    print(classification_report(real_val_y, np.where(real_val_pred_y > online_threshold, 1, 0), digits=6))

    # 测试不同的阈值组合
    max_f1 = 0
    max_threshold_in = 0
    max_threshold_not_in = 0
    for threshold_in in np.arange(0.30, 0.60, 0.01):
        for threshold_not_in in np.arange(0.30, 0.60, 0.01):
            pred_in = np.where(real_val_pred_in > threshold_in, 1, 0)
            pred_not_in = np.where(real_val_pred_not_in > threshold_not_in, 1, 0)
            pred = np.concatenate([pred_in, pred_not_in])
            tmp_f1 = f1_score(real_val_y, pred)
            #             print(threshold_in,' ',threshold_not_in , ' f1 score: ', tmp_f1)
            if tmp_f1 > max_f1:
                print(threshold_in, ' ', threshold_not_in, ' f1 score: ', tmp_f1)
                max_f1 = tmp_f1
                max_threshold_in = threshold_in
                max_threshold_not_in = threshold_not_in
    print('best threshold for real val: ', max_threshold_in, ' ', max_threshold_not_in, '  f1 :', max_f1)
    pred_results.append(real_val_pred_y)

# 对五次模型预测的结果求平均
res = np.mean(np.vstack(pred_results), axis=0)

print("different threshold for real val pred in")
max_f1 = 0
max_threshold_in = 0
for threshold in np.arange(0.30, 0.60, 0.01):
    tmp_f1 = f1_score(real_val_y_in, np.where(res[:len_of_in] > threshold, 1, 0))
    print(threshold, ' f1 score: ', tmp_f1)
    if tmp_f1 > max_f1:
        max_f1 = tmp_f1
        max_threshold_in = threshold
print('best threshold: ', max_threshold_in, '  f1 :', max_f1)

print("different threshold for real val pred not in")
max_f1 = 0
max_threshold_not_in = 0
for threshold in np.arange(0.30, 0.60, 0.01):
    tmp_f1 = f1_score(real_val_y_not_in, np.where(res[len_of_in:] > threshold, 1, 0))
    print(threshold, ' f1 score: ', tmp_f1)
    if tmp_f1 > max_f1:
        max_f1 = tmp_f1
        max_threshold_not_in = threshold
print('best threshold: ', max_threshold_not_in, '  f1 :', max_f1)

# 测试不同的阈值组合
max_f1 = 0
max_threshold_in = 0
max_threshold_not_in = 0
for threshold_in in np.arange(0.30, 0.60, 0.01):
    for threshold_not_in in np.arange(0.30, 0.60, 0.01):
        pred_in = np.where(res[:len_of_in] > threshold_in, 1, 0)
        pred_not_in = np.where(res[len_of_in:] > threshold_not_in, 1, 0)
        pred = np.concatenate([pred_in, pred_not_in])
        tmp_f1 = f1_score(real_val_y, pred)
        #         print(threshold_in,' ',threshold_not_in , ' f1 score: ', tmp_f1)
        if tmp_f1 > max_f1:
            print(threshold_in, ' ', threshold_not_in, ' f1 score: ', tmp_f1)
            max_f1 = tmp_f1
            max_threshold_in = threshold_in
            max_threshold_not_in = threshold_not_in
print('best threshold for total piece: ', max_threshold_in, ' ', max_threshold_not_in, '  f1 :', max_f1)

submission = pd.DataFrame()
submission['predicted_score'] = res
submission['label'] = submission['predicted_score'].apply(lambda x: 1 if x > online_threshold else 0)
submission.to_csv(
    'pred_val_by_model.csv', sep=',', index=False)