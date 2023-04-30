from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import naive_bayes
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = naive_bayes.NaiveBayes()
model.fit(X_train, y_train)
print(model.predict([5.0, 3.5, 1.4, 0.2]))
print(model.predict([7, 2.2, 4.7, 1.4]))
print(model.predict([5.8, 3.6, 5.1, 1.8]))
# for x in X_test:
#     print(model.predict(x))
print(f"准确率：{model.score(X_test, y_test)}")
