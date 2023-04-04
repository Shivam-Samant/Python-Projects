# Uni varient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report

df = pd.read_csv('./CSV/insurance_data.csv')
print(df.describe())

plt.scatter(df.age, df.bought_insurance, color="red", marker='+')
plt.show()

print(df.info())
print(df.isnull().sum())

model = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.2)

model.fit(x_train, y_train)

y_predict = model.predict(x_train)

print(y_predict)

mae = mean_absolute_error(y_train, y_predict)
mse = mean_squared_error(y_train, y_predict)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

# print("Train")
print(model.score(x_train, y_predict))
print(model.predict_proba(x_train))

# print("Testing")
y_pred = model.predict(x_test)

print(model.score(x_test, y_test))
print(model.predict_proba(x_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Not purchased", "purchased"]))
