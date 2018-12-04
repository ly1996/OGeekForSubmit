from gensim.models import KeyedVectors
import path_file
import numpy as np
from gensim import matutils
import json
import jieba

def get_stop_words():
    stop_wrods_name = path_file.stop_words_name
    _stop_words_list = list()
    with open(stop_wrods_name, encoding='utf-8') as f:
        for line in f:
            _stop_words_list.append(line.strip())

    _stop_words_set = set(_stop_words_list)
    return _stop_words_set

stop_words_set = get_stop_words()

def char_list_cheaner(char_list):
    new_char_list = list()
    for char in char_list:
        if len(char) == 0:
            continue
        if char in stop_words_set:
            continue
        new_char_list.append(char)

    return new_char_list

#构造分词和向量之间的字典
print ("load vec")
vec_txt = path_file.vec_txt
vecdic = KeyedVectors.load_word2vec_format(vec_txt, binary=False)
vec_len = len(vecdic['重庆'].tolist())
print ("vec_len :",vec_len)

# file1 = open('0_200000.json', 'r')
# js1 = file1.read()
# sentence_to_cut1 = json.loads(js1)
# file1.close()
#
# file1 = open('200000_400000.json', 'r')
# js1 = file1.read()
# sentence_to_cut2 = json.loads(js1)
# file1.close()
#
# file1 = open('400000_600000.json', 'r')
# js1 = file1.read()
# sentence_to_cut3 = json.loads(js1)
# file1.close()
#
# file1 = open('600000_800000.json', 'r')
# js1 = file1.read()
# sentence_to_cut4 = json.loads(js1)
# file1.close()
#
# file1 = open('800000_1000000.json', 'r')
# js1 = file1.read()
# sentence_to_cut5 = json.loads(js1)
# file1.close()
# file1 = open('1000000_1200000.json', 'r')
# js1 = file1.read()
# sentence_to_cut6 = json.loads(js1)
# file1.close()
#
# file1 = open('1200000_1400000.json', 'r')
# js1 = file1.read()
# sentence_to_cut7 = json.loads(js1)
# file1.close()
# file1 = open('1400000_1600000.json', 'r')
# js1 = file1.read()
# sentence_to_cut8 = json.loads(js1)
# file1.close()
#
# file1 = open('1600000_1800000.json', 'r')
# js1 = file1.read()
# sentence_to_cut9 = json.loads(js1)
# file1.close()
# file1 = open('1800000_2000000.json', 'r')
# js1 = file1.read()
# sentence_to_cut10 = json.loads(js1)
# file1.close()
#
# file1 = open('2000000_2250000.json', 'r')
# js1 = file1.read()
# sentence_to_cut11 = json.loads(js1)
# file1.close()

# text_data_cut_dict = {
#     **sentence_to_cut1,
#     **sentence_to_cut2,
#     **sentence_to_cut3,
#     **sentence_to_cut4,
#     **sentence_to_cut5,
#     **sentence_to_cut6,
#     **sentence_to_cut7,
#     **sentence_to_cut8,
#     **sentence_to_cut9,
#     **sentence_to_cut10,
#     **sentence_to_cut11
# }

# import thulac
# thu1 = thulac.thulac()  #默认模式

def sentence_vec(str):
    str = str.lower()
    # cut = thu1.cut(str, text=False)
    items = list(jieba.cut(str))
    # try:
    #     items = text_data_cut_dict[str]
    # except KeyError:
    #     cut = thu1.cut(str, text=False)
    #     items = []
    #     for item in cut:
    #         items.append(item[0])
    # items = []
    # for item in cut:
    #     items.append(item[0])

    # items = char_list_cheaner(items)
    sentence = []
    for item in items:
        try:
            word_vec = vecdic[item]
        except KeyError:
            continue
        else:
            sentence.append(word_vec)

    if len(sentence) > 0:
        item_list = matutils.unitvec(np.array(sentence).mean(axis=0))
        return item_list
    else:
        # print ("w2v len == 0",str)
        return np.zeros(vec_len)

