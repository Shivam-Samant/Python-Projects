import pickle

# Open the model
fp = open("./Models/LinearRegressionModel.obj", "rb")

# Load the model
model = pickle.load(fp)

# Predict the price of a house with 3300 sqft area
print(model.predict([[3300]]))
