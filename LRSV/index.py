import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import os

# Read the data
df = pd.read_csv('./CSV/HousePrediction.csv')

print(df, df['price'])

# Plot the data
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

# Create a linear regression model
model = linear_model.LinearRegression()

# Train the model
model.fit(df[['area']], df.price)

# Predict the price of a house with 3300 sqft area
print(model.predict([[3300]]))

# coefficient
coefficient = model.coef_
print(coefficient)

# intercept
intercept = model.intercept_
print(intercept)

# y = mx + c
# price = m * area + c
print(coefficient * 3300 + intercept)

# Create the 'Models' directory if it does not exist
if not os.path.exists('./Models'):
    os.makedirs('./Models')

# Save the model
with open('./Models/LinearRegressionModel.obj', 'wb') as f:
    pickle.dump(model, f)
