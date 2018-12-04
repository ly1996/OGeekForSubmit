from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

#数据路径
import path_file

train_txt = path_file.train_txt
val_txt = path_file.val_txt

train_data = pd.read_table(train_txt,
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8',quoting=3).astype(str)
val_data = pd.read_table(val_txt,
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8',quoting=3).astype(str)
#去噪，只有一条数据
train_data = train_data[train_data['label'].isin(['0', '1'])]

train_data['prefix_title'] = train_data['prefix'].apply(lambda x: x + "_")
train_data['prefix_title'] = train_data['prefix_title'] + train_data['title']
val_data['prefix_title'] = val_data['prefix'].apply(lambda x: x + "_")
val_data['prefix_title'] = val_data['prefix_title'] + val_data['title']

prefixs = train_data.groupby(['prefix'], as_index=False)['prefix'].agg({'cnt': 'count'})['prefix'].tolist()
print('prefixs : ', len(prefixs))

prefixs_title = train_data.groupby(['prefix_title'], as_index=False)['prefix_title'].agg({'cnt': 'count'})['prefix_title'].tolist()

print (len(prefixs_title))

val_feat = val_data.copy()

# val model 1
val_feat_in = val_feat[val_feat['prefix'].isin(prefixs)]
# val model2
val_feat_not_in = val_feat[~val_feat['prefix'].isin(prefixs)]

print("len of val in prefixs : ", val_feat_in.shape[0])
print("len of val not in prefixs : ", val_feat_not_in.shape[0])

# val model 1
val_feat_in = val_feat[val_feat['prefix_title'].isin(prefixs_title)]
# val model2
val_feat_not_in = val_feat[~val_feat['prefix_title'].isin(prefixs_title)]

print("len of val in prefixs : ", val_feat_in.shape[0])
print("len of val not in prefixs : ", val_feat_not_in.shape[0])




