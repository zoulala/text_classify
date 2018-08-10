
import tensorflow as tf
from read_utils import TextConverter
from model import Model, Config
import os



if __name__=="__main__":
    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    train_files = '../data/cnews.train.txt'
    val_files = '../data/cnews.val.txt'
    test_files = '../data/cnews.test.txt'
    save_file = 'cnews.vocab_label.pkl'

    # 数据处理
    converter = TextConverter(train_files, save_file, max_vocab=Config.vocab_size, seq_length=Config.seq_length)
    print('vocab size:',converter.vocab_size)
    print('labels:',converter.label)

    test_texts, test_labels = converter.load_data(test_files)
    test_x, test_x_len, test_y = converter.texts_to_arr(test_texts, test_labels)

    test_g = converter.val_samples_generator(test_x, test_x_len, test_y, Config.batch_size)

    model = Model(Config)

    # 加载上一次保存的模型
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to testing...')
    model.test(test_g)



