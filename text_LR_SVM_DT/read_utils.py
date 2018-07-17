import os
import pickle
import numpy as np
from collections import Counter

class TextConverter(object):

    def __init__(self, train_dir=None, save_dir=None,  max_vocab=5000 ):
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                self.vocab, self.label = pickle.load(f)
        else:
            self.build_vocab(train_dir, save_dir, max_vocab)

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.label_to_int_table = {c: i for i, c in enumerate(self.label)}

    def load_data(self, filename):
        contents, labels = [], []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                label, content = line.strip().split('\t')
                contents.append(content)
                labels.append(label)
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
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def texts_to_arr(self, texts, labels):
        return np.array([self.text_to_arr(text) for text in texts]), np.array([self.label_to_int(label) for label in labels])




if __name__=="__main__":
    base_dir = '../data'
    train_files = os.path.join(base_dir, 'cnews.train.txt')
    save_file = 'cnews.vocab.pkl'

    converter = TextConverter(train_files, save_file, max_vocab=5000)
    print(converter.vocab_size)