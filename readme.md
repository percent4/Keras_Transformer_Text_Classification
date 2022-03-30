本项目使用Keras实现Transformer模型来进行文本分类（中文、英文均支持）。

### 数据集

数据集位于src目录下的data目录。

- imdb.npz: 英语影视评论数据集
- sougou_mini: 搜狗分类数据集，共5个类别
- weibo_sentiment: 微博情绪分类数据集，共7个类别


### 模型结果

|数据集|次数|模型参数|accuracy|marco avg F1|
|---|---|---|---|---|
|imdb|1|epoch: 5, maxlen: 128, batch_size: 32|0.8406|0.8406|
|imdb|2|epoch: 5, maxlen: 100, batch_size: 32|0.8258|0.8257|
|imdb|3|epoch: 5, maxlen: 80, batch_size: 32|0.8154|0.8153|
|sougou_mini|1|epoch: 10, maxlen: 300, batch_size: 16|||
|sougou_mini|2|epoch: 10, maxlen: 256, batch_size: 16|||
|sougou_mini|3|epoch: 10, maxlen: 200, batch_size: 16|0.9293|0.9287|
|weibo_sentiment|1|epoch: 10, maxlen: 180, batch_size: 32|0.6011|0.6093|
|weibo_sentiment|2|epoch: 10, maxlen: 128, batch_size: 32|0.6280|0.6325|
|weibo_sentiment|3|epoch: 10, maxlen: 100, batch_size: 32|0.6215|0.6235|

