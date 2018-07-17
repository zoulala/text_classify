from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

class Config(object):
    """CNN配置参数"""
    file_name = 'cnn1'  #保存模型文件

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    train_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

        # cnn模型
        self.cnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
            embedding = tf.concat([embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            one_hot_y = tf.one_hot(self.input_y, self.config.num_classes)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.input_y, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for batch_x, batch_x_len, batc_y in batch_train_g:
                start = time.time()
                feed = {self.input_x: batch_x,
                        self.input_y: batc_y,
                        self.keep_prob: self.config.train_keep_prob}

                _, batch_loss, acc, y_pre= sess.run([self.optim, self.loss, self.acc, self.y_pred_cls], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    # self.saver.save(sess, os.path.join(save_path, 'model'), global_step=self.global_step)
                    y_pres = np.array([])
                    accs = np.array([])
                    for batch_x, batch_x_len, batc_y in val_g:
                        feed = {self.input_x: batch_x,
                                self.input_y: batc_y,
                                self.keep_prob: 1}

                        y_pre, acc = sess.run([self.y_pred_cls, self.acc], feed_dict=feed)
                        y_pres = np.append(y_pres, y_pre)
                        accs = np.append(accs, acc)

                    # 计算预测准确率
                    print('val lens:',len(y_pres))
                    print("val accuracy:{:.2f}%... ".format(accs.mean() * 100),
                            'best:{:.4f}'.format(self.global_loss.eval())*100)
                    acc_val = accs.mean()
                    if acc_val > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc_val)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)

                if self.global_step.eval() >= self.config.max_steps:
                    break


    def test(self, test_g ):
        with self.session as sess:
            y_pres = np.array([])
            accs = np.array([])
            for batch_x, batch_x_len, batc_y in test_g:
                feed = {self.input_x: batch_x,
                        self.input_y: batc_y,
                        self.keep_prob: 1}

                y_pre, acc = sess.run([self.y_pred_cls, self.acc], feed_dict=feed)
                y_pres = np.append(y_pres, y_pre)
                accs = np.append(accs, acc)
                print('...............................................................')

            # 计算预测准确率
            print('test lens:', len(y_pres))
            print("test accuracy:{:.2f}%... ".format(accs.mean() * 100))




