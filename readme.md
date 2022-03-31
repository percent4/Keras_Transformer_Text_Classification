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
|sougou_mini|1|epoch: 10, maxlen: 300, batch_size: 16|0.9455|0.9449|
|sougou_mini|2|epoch: 10, maxlen: 256, batch_size: 16|0.9475|0.9470|
|sougou_mini|3|epoch: 10, maxlen: 200, batch_size: 16|0.9293|0.9287|
|weibo_sentiment|1|epoch: 10, maxlen: 180, batch_size: 32|0.6011|0.6093|
|weibo_sentiment|2|epoch: 10, maxlen: 128, batch_size: 32|0.6280|0.6325|
|weibo_sentiment|3|epoch: 10, maxlen: 100, batch_size: 32|0.6215|0.6235|

### 模型预测

搜狗小分类数据预测结果:


>text: 近日，武警第二机动总队某支队紧抓春季练兵黄金期，组织官兵进行多课目实弹射击考核，全面检验官兵实弹射击水平。

>label: 军事

>text:  进入到2022年2月份，零跑汽车的表现依旧“突飞猛进”，2022年2月份，零跑汽车销量3435台，同比增长447%。2022年1-2月，零跑汽车销售11520台，月销量同比实现连续11个月超200%增速，稳居造车新势力二线阵营。

>label: 汽车

>text: 本报讯（记者 孙军）“近3年来，青岛市投资10.39亿元用于更新升级教育信息化基础设施，实现了校园无线网络100％覆盖，形成了支撑教育高质量发展的‘高速路网’。”青岛市委常委、宣传部长、市委教育工委书记孙立杰日前在接受本报记者采访时表示，青岛正积极推动“数字青岛”建设，把教育信息化作为优质教育资源倍增的重要助力，全面支撑教育现代化进程。

>label: 教育

