# -*- coding: utf-8 -*-
# @Time : 2022/3/31 0:13
# @Author : Jclian91
# @File : get_npz_file.py
# @Place : Minghang, Shanghai
import json
import pandas as pd
import numpy as np

labels = ['体育', '健康', '军事', '教育', '汽车']
label_dict = dict(zip(labels, range(len(labels))))

# 获取字符集并写入json文件中
char_set = set()
train_df = pd.read_csv('train.csv').fillna('')
test_df = pd.read_csv('test.csv').fillna('')
for index, row in train_df.iterrows():
    for char in row['content']:
        char_set.add(char)
for index, row in test_df.iterrows():
    for char in row['content']:
        char_set.add(char)

char_dict = dict(zip(list(char_set), range(len(char_set))))

with open('char_dict.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(char_dict, ensure_ascii=False, indent=4))

# 将训练集合测试集写入至npz文件中
x_train = []
y_train = []
x_test = []
y_test = []
for index, row in train_df.iterrows():
    y_train.append(label_dict[row['label']])
    x_train.append([char_dict[char] for char in row['content']])
for index, row in test_df.iterrows():
    y_test.append(label_dict[row['label']])
    x_test.append([char_dict[char] for char in row['content']])

np.savez('sougou_mini.npz',
         x_train=np.array(x_train),
         y_train=np.array(y_train),
         x_test=np.array(x_test),
         y_test=np.array(y_test))
