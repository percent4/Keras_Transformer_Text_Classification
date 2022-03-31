# -*- coding: utf-8 -*-
# @Time : 2022/3/30 22:53
# @Author : Jclian91
# @File : chinese_model_train.py
# @Place : Minghang, Shanghai
import numpy as np
from load_data import data_loader
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, Dense
from sklearn.metrics import classification_report

from model import TransformerBlock


# 创建分类模型
def create_cls_model(features):
    inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(features, 128)(inputs)
    O_seq = TransformerBlock(8, 16, 128)(embeddings)
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(5, activation='sigmoid')(O_seq)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# 模型训练主函数
if __name__ == '__main__':
    max_features = 5500
    max_len = 300
    batch_size = 16

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = data_loader(path='./data/sougou_mini/sougou_mini.npz', num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = create_cls_model(max_features)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=(x_test, y_test))
    # 模型保存
    model.save_weights('./data/sougou_mini/sougou_mini.h5')

    # 模型评估
    print('Evaluate...')
    result = model.predict(x_test)
    y_predict = np.argmax(result, axis=1)
    print(classification_report(y_true=y_test.tolist(), y_pred=y_predict.tolist(), digits=4))
