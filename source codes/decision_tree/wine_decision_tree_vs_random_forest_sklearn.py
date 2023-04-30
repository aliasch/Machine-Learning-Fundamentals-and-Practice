import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model_tree = DecisionTreeClassifier(random_state=0)
model_forest = RandomForestClassifier(random_state=0)
model_tree = model_tree.fit(X_train, y_train)
model_forest = model_forest.fit(X_train, y_train)
score_c = model_tree.score(X_test, y_test)
score_r = model_forest.score(X_test, y_test)
print(f"单棵决策树: {score_c}, 随机森林: {score_r}")
# 交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
rfc_l = []  # 随机森林列表
clf_l = []  # 决策树列表
for i in range(10):
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, X, y, cv=10).mean()
    clf_l.append(clf_s)
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, X, y, cv=10).mean()
    rfc_l.append(rfc_s)
plt.plot(range(1, 11), rfc_l, label="随机森林")
plt.plot(range(1, 11), clf_l, label="决策树")
plt.xlabel('比较次数')
plt.ylabel('对应分数')
plt.legend()
plt.show()
scores = []
for i in range(50):
    rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
    rfc_s = cross_val_score(rfc, X, y, cv=10).mean()
    scores.append(rfc_s)
print(max(scores), scores.index(max(scores)))
plt.plot(range(1, 51), scores)
plt.xlabel('森林中树的数量')
plt.ylabel('对应分数')
plt.show()
