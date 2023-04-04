import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("\nLinearRegression Model\n")

df = pd.read_csv('./CSV/filteredHousingData.csv')

plt.scatter(df.CRIM, df.MEDV) # (partial -ve relation)
plt.show()

plt.scatter(df.ZN, df.MEDV) # remove
plt.show()

plt.scatter(df.INDUS, df.MEDV) # remove (-ve relation)
plt.show()

plt.scatter(df.CHAS, df.MEDV) # delete
plt.show()

plt.scatter(df.NOX, df.MEDV) # (-ve relation)
plt.show()

plt.scatter(df.RM, df.MEDV) # (+ve relation)
plt.show()

plt.scatter(df.AGE, df.MEDV) # (-ve relation)
plt.show()

plt.scatter(df.DIS, df.MEDV) # (+ve relation)
plt.show()

plt.scatter(df.RAD, df.MEDV) # delete
plt.show()

plt.scatter(df.TAX, df.MEDV) # partial dependent
plt.show()

plt.scatter(df.PTRATIO, df.MEDV) # partial dependent
plt.show()

plt.scatter(df.B, df.MEDV) # remove
plt.show()

plt.scatter(df.LSTAT, df.MEDV) # (-ve relation)
plt.show()


model = LinearRegression()

filterDf = df.drop(['CHAS', 'ZN', 'RAD', 'B', 'CRIM', 'PTRATIO', 'TAX',  'MEDV'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(filterDf, df.MEDV, test_size=0.2)

model.fit(x_train, y_train)

predicetedMedianValue = model.predict(x_test)

# print(predicetedMedianValue)

mae = mean_absolute_error(y_test, predicetedMedianValue)
mse = mean_squared_error(y_test, predicetedMedianValue)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

plt.scatter(y_test, predicetedMedianValue)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

print("\nRandomForestRegressor Model\n")

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets 
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print('MAE:',mean_absolute_error(y_test, y_pred))
print('MSE:',mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

