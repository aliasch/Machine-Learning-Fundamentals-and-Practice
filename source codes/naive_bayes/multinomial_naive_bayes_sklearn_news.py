from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf
from sklearn.naive_bayes import MultinomialNB

news = fetch_20newsgroups(subset='all')
# 进行训练集和测试集切分
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
# 对数据集进行特征提取
# print(news.data[0])
tf = TfidfVectorizer()
X_train = tf.fit_transform(X_train)
X_test = tf.transform(X_test)
print(tf.get_feature_names())
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# print(y_predict)
# 得出准确率
print(f"准确率为：{model.score(X_test, y_test)}")
