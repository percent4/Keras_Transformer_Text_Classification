# -*- coding: utf-8 -*-
# @Time : 2022/3/30 22:53
# @Author : Jclian91
# @File : imdb_model_train.py
# @Place : Minghang, Shanghai
import numpy as np
from load_data import data_loader
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, Dense
from sklearn.metrics import classification_report

from model import TransformerBlock


# 模型训练主函数
if __name__ == '__main__':
    max_features = 20000
    max_len = 80
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = data_loader(path='./data/imdb/imdb.npz', num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(max_features, 128)(inputs)
    O_seq = TransformerBlock(8, 16, 128)(embeddings)
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(1, activation='sigmoid')(O_seq)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test))

    # 模型评估
    print('Evaluate...')
    result = model.predict(x_test)
    y_predict = [1 if _[0] >= 0.5 else 0 for _ in result.tolist()]
    print(classification_report(y_true=y_test.tolist(), y_pred=y_predict, digits=4))
