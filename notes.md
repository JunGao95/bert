## Notes of **BERT**

- 使用FullTokenizer对技术需求相关自然语言的解析结果实例：

        tokenizer = FullTokenizer(vocab_file='chinese_L-12_H-768_A-12\\vocab.txt')
        tokenizer.tokenize('设计时速应达到360km/h')
        #['设', '计', '时', '速', '应', '达', '到', '360', '##km', '/', 'h']
        tokenizer.tokenize('Deep Learning, BERT, CNN, RNN, LSTM')
        #['deep', 'learning', ',', 'be', '##rt', ',', 'cnn', ',', 'rn', '##n', ',', 'l', '##st', '##m']

## Notes of bert-ner.py

- 使用bert-ner.py，指定路径时，使用data_dir下train/eval/test.txt的文件结构；

- lines = [[l1, w1], [l1, w2], ..., [ln, wn]], l/w分别是一句话的label和word列表；

- examples是一个InputExample的列表，每一个InputExample有三个属性，分别是**guid(train-0),text和label**;

- input_mask是为了表征是Padding的文本还是正常文本设定的特征，如果是Padding文本则置0，否则为1；

- 如需更换任务，需要在434-436行更改评价标准，此处评价标准是将所有命名实体认为是正例，其余认为是反例；

- 进行实体标注时，[CLS]开头，[SEP]结尾，非实体标注为O，分词后后半词标注为X；



## ***Questions***

- 218-226行，为什么只标注前词为对应标签，后面用“X”？

- Q:为什么在获取num_labels时要在列表长度基础上加1？

  A:由于210行令1-12这十二个数字分别代表定义好的12个标签，0代表的是填充内容；

- Q:为什么输出txt时未包含有0的Padding部分？

  A:602行将所有输出为0的删去了。

- Q:tensorflow里的shape(n, )和(n, 1)到底是什么区别？

  A:


        