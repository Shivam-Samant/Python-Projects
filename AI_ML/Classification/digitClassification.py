from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report


digits = load_digits()
print(dir(digits))
print(len(digits))
print(digits.data[1])
print(digits.target_names)

model = LogisticRegression(max_iter=5000)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(y_predict)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

print(confusion_matrix(y_test, y_predict))

print(classification_report(y_test, y_predict))