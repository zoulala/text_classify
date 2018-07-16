from sklearn import svm,tree,linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from data_process import TextConverter

train_files = '../data/cnews.train.txt'
test_files = '../data/cnews.test.txt'
save_file = 'cnews.vocab_label.pkl'

converter = TextConverter(train_files, save_file, max_vocab=5000)
print(converter.vocab_size)
print(converter.label)

train_texts, train_labels = converter.load_data(train_files)
# train_x, train_y = converter.texts_to_arr(train_texts, train_labels)

val_texts, val_labels = converter.load_data(train_files)
# val_x, val_y = converter.texts_to_arr(train_texts, train_labels)

test_texts, test_labels = converter.load_data(test_files)
# test_x, test_y = converter.texts_to_arr(test_texts, test_labels)

# -------------feature extract --------------------
vec = TfidfVectorizer(ngram_range=(1,1),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1, token_pattern=r"(?u)\w")


train_features = vec.fit_transform(train_texts).toarray()
val_features = vec.transform(val_texts).toarray()
test_features = vec.transform(test_texts).toarray()


##  -------------- SVM -------------
# lin_clf = svm.LinearSVC()  ### test accuracy:0.952


## ------------ DT -----------
lin_clf = tree.DecisionTreeClassifier()  # ## test accuracy:0.8057

## ------------- LR ----------
# lin_clf = linear_model.LogisticRegression()  ### test accuracy:0.9416


lin_clf.fit(train_features,train_labels)
val_pre = lin_clf.predict(val_features)
print("val accuracy score:",(val_pre==val_labels).mean())

test_pre = lin_clf.predict(test_features)
print("test accuracy score:",(test_pre==test_labels).mean())






