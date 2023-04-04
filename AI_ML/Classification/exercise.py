import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report

df = pd.read_csv('./CSV/HR_comma_sep.csv')

plt.plot(df.satisfaction_level, df.salary)
plt.xlabel('Salary')
plt.ylabel('Satisfaction level')
plt.title('Satisfaction level vs Salary')
plt.show()

df['salary'].replace({'low': 1, 'medium': 2, 'high': 3}, inplace=True)

features = df.drop(['Work_accident', 'left', 'promotion_last_5years', 'Department', 'salary'], axis=1)

model = LogisticRegression(max_iter=1000)

x_train, x_test, y_train, y_test = train_test_split(features, df.salary, test_size=0.2)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(y_predict)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company
print(model.predict([[0.8, 0.53, 2, 157, 3]]))

print(model.score(x_test, y_test))
