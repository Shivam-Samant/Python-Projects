import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


# loading iris data
iris = load_iris()

print(dir(iris))
print(iris.target_names, iris.feature_names)

# create dataframes
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
print(df.shape)

# appending target and flower names
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

# data slicing
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


X = df.drop(['target', 'flower_name'], axis=1)
y = df.target

model = SVC()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predict = model.predict(x_test)
print(y_predict)


print(confusion_matrix(y_test, y_predict))

print(classification_report(y_test, y_predict, target_names=["setosa", "versicolor", "virginica"]))


