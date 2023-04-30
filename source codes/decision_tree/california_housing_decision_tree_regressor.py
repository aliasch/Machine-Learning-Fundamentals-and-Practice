from sklearn.model_selection import train_test_split
import decision_tree.decision_tree_regressor as tree
from sklearn.datasets import fetch_california_housing

# 读取加州房价数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
# 决策树回归
model = tree.DecisionTreeRegressor(max_depth=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
