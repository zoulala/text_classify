import os
import random
import pickle
import numpy as np
from collections import Counter

class TextConverter(object):

    def __init__(self, train_dir=None, save_dir=None,  max_vocab=5000 , seq_length = 1000):
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                self.vocab, self.label = pickle.load(f)
        else:
            self.build_vocab(train_dir, save_dir, max_vocab)

        self.seq_length = seq_length  # 样本序列最大长度
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.label_to_int_table = {c: i for i, c in enumerate(self.label)}

    def load_data(self, filename):
        contents, labels = [], []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                label, content = line.strip().split('\t')
                contents.append(content)
                labels.append(label)
        cc = list(zip(contents, labels))
        random.shuffle(cc)
        contents, labels = zip(*cc)
        return contents,labels

    def build_vocab(self, train_dir, vocab_dir, vocab_size=None):
        """根据训练集构建词汇表，存储"""

        contents, labels = self.load_data(train_dir)

        self.label = list(set(labels))

        all_data = []
        for content in contents:
            all_data.extend(content)
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size)  # 数量排序[（word1, num1）,(word2,num2),...]
        self.vocab, _ = list(zip(*count_pairs))
        with open(vocab_dir, 'wb') as f:
            pickle.dump((self.vocab, self.label), f)

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def label_to_int(self, label):
        if label in self.label_to_int_table:
            return self.label_to_int_table[label]
        else:
            raise ("label not in train's list !")

    def text_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length:
            arr += [last_num] * (self.seq_length - query_len)
        else:
            arr = arr[:self.seq_length]
            query_len = self.seq_length

        return np.array(arr), query_len



    def texts_to_arr(self, texts, labels):
        texts_arr = []
        texts_len = []
        for text in texts:
            t_arr, t_len = self.text_to_arr(text)
            texts_arr.append(t_arr)
            texts_len.append(t_len)

        return np.array(texts_arr), np.array(texts_len), np.array([self.label_to_int(label) for label in labels],dtype=int)

    def batch_generator(self,train_x, train_x_len, train_y, batchsize):
        '''产生训练batch样本'''
        assert len(train_x)==len(train_x_len)==len(train_y), "error:len(x)!=len(y)."

        n_samples = len(train_x)
        n_batches = int(n_samples / batchsize)
        print("batches:",n_batches)
        n = n_batches * batchsize
        while True:
            for i in range(0, n, batchsize):
                batch_x = train_x[i:i + batchsize]
                batch_x_len = train_x_len[i:i + batchsize]
                batc_y = train_y[i:i + batchsize]
                yield batch_x, batch_x_len, batc_y

    def val_samples_generator(self,val_x, val_x_len, val_y, batchsize=500):
        '''产生验证样本，batchsize分批验证，减少运行内存'''
        assert len(val_x) == len(val_x_len) == len(val_y), "error:len(x)!=len(y)."
        val_g = []
        n = len(val_x)

        for i in range(0,n,batchsize):
            batch_x = val_x[i:i + batchsize]
            batch_x_len = val_x_len[i:i + batchsize]
            batc_y = val_y[i:i + batchsize]
            val_g.append((batch_x, batch_x_len, batc_y))
        return val_g




if __name__=="__main__":
    base_dir = '../data'
    train_files = os.path.join(base_dir, 'cnews.train.txt')
    save_file = 'cnews.vocab.pkl'

    converter = TextConverter(train_files, save_file, max_vocab=5000)
    print(converter.vocab_size)