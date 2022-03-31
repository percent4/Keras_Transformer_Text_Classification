# -*- coding: utf-8 -*-
# @Time : 2022/3/31 12:29
# @Author : Jclian91
# @File : chinese_model_predict.py
# @Place : Minghang, Shanghai
import json
import numpy as np

from load_data import data_loader
from chinese_model_train import create_cls_model
from keras.preprocessing.sequence import pad_sequences


# 加载模型
def load_model(features, model_path):
    model = create_cls_model(features)
    model.load_weights(model_path)
    return model


# 文本分类预测
def text_predict(text, labels, features, model, max_len):
    with open('./data/sougou_mini/char_dict.json', 'r', encoding='utf-8') as f:
        char_dict = json.loads(f.read())
    x_train = [[char_dict.get(char, 2) for char in text]]
    y_train = [0]
    x_test, y_test = x_train, y_train
    np.savez('./data/sougou_mini/test.npz',
             x_train=np.array(x_train),
             y_train=np.array(y_train),
             x_test=np.array(x_test),
             y_test=np.array(y_test))
    (x_train, y_train), (x_test, y_test) = data_loader(path='./data/sougou_mini/test.npz',
                                                       num_words=features)
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
    result = model.predict(x_train)
    y_predict = np.argmax(result, axis=1)
    label = labels[y_predict.tolist()[0]]
    return label


if __name__ == '__main__':
    max_features = 5500
    max_length = 300
    model_file_path = './data/sougou_mini/sougou_mini.h5'
    cls_model = load_model(max_features, model_file_path)
    my_text = '本报讯（记者 孙军）“近3年来，青岛市投资10.39亿元用于更新升级教育信息化基础设施，实现了校园无线网络100％覆盖，' \
              '形成了支撑教育高质量发展的‘高速路网’。”青岛市委常委、宣传部长、市委教育工委书记孙立杰日前在接受本报记者采访时表示，' \
              '青岛正积极推动“数字青岛”建设，把教育信息化作为优质教育资源倍增的重要助力，全面支撑教育现代化进程。'
    label = text_predict(text=my_text,
                         labels=['体育', '健康', '军事', '教育', '汽车'],
                         features=max_features,
                         model=cls_model,
                         max_len=max_length)
    print(f'text: {my_text}\nlabel: {label}')
