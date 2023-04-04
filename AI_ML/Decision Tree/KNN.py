import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./CSV/salaries.csv')
print(df.head())
print(df.isnull().sum())
print(df.nunique())

# data = pd.get_dummies(df)
# OR
le = LabelEncoder()

ar_company = le.fit_transform(df.company)
ar_job = le.fit_transform(df.job)
ar_degree = le.fit_transform(df.degree)
df = df.drop(['company', 'job', 'degree'], axis=1)

df_company = pd.Series(ar_company, name="company")
df_job = pd.Series(ar_job, name="job")
df_degree = pd.Series(ar_degree, name="degree")

df = pd.concat([df, df_company, df_job, df_degree], axis=1)
# print(df.head())

x = df.drop(['salary_more_then_100k'], axis=1)
y = df.salary_more_then_100k

print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

y_predict = model.predict(x_train)

# print(y_predict)

mae = mean_absolute_error(y_train, y_predict)
mse = mean_squared_error(y_train, y_predict)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

# print("Testing")
y_pred = model.predict(x_test)
print("score", model.score(x_test, y_pred))
print("predict_proba", model.predict_proba(x_test))

print("score", model.score(x_test, y_test))

print("confusion_matrix\n", confusion_matrix(y_test, y_pred))

print("classification_report\n", classification_report(y_test, y_pred))


