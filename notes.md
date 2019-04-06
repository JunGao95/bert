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

- ***218-226行，为什么只标注前词为对应标签，后面用“X”？***

- 
        