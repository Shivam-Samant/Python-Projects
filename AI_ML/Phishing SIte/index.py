import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

# Loading data from CSV file
df = pd.read_csv('./Phishing Site/CSV/Website Phishing.csv')

# printing first 5 rows
print(df.head().to_string())

# Checking null values
print(df.isnull().any())

# Checking na values
print(df.isna().any())

# Displaying the shape of data
print("Shape of data:", df.shape)

# Describing the data
print("Data description:\n", df.describe().to_string())

# compute correlation matrix
corr_matrix = df.corr()
print("CORR MATRIX", corr_matrix)

sns.boxplot(df)
plt.show()

p1 = df[df['popUpWindow'] == 1] # 110
p0 = df[df['popUpWindow'] == 0] # 298
pn1 = df[df['popUpWindow'] == -1] # 316

print(p1.value_counts(), p0.value_counts(), pn1.value_counts())

# plot heatmap of correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Defining a function to plot scatter plots
def plot_scatter(feature, target, featureLabel, targetLabel, title):
    plt.scatter(feature, target)
    plt.xlabel(featureLabel)
    plt.ylabel(targetLabel)
    plt.title(title)
    plt.show()

# Plotting scatter plots for different features
features = df.columns[:-1]
for feature in features:
    plot_scatter(df[feature], df['Result'], feature, 'Result', feature + ' vs Result')

# feature selection
features = df.drop(['Result', 'having_IP_Address'], axis=1) # features
target = df.Result # target

# splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)

# Defining a function to train and evaluate models
def model_evaluation(model):
    # training the model
    model.fit(x_train, y_train)
    # Evaluating the model's performance
    print(model.__class__.__name__ + " Model Evaluation:")
    print("TRAINING SET")
    # Making predictions on the training set
    y_pred = model.predict(x_train)
    print("Accuracy:", model.score(x_train, y_train))
    print("Confusion matrix\n", confusion_matrix(y_train, y_pred))
    print("Classification report\n", classification_report(y_train, y_pred, target_names=["Phishy", "Suspecious", "Legitimate"]))

    print("\nTESTING SET")
    # Making predictions on the test set
    y_pred = model.predict(x_test)
    print("Accuracy:", round(model.score(x_test, y_test)*100, 2), "%")
    print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
    print("Classification report\n", classification_report(y_test, y_pred, target_names=["Phishy", "Suspecious", "Legitimate"]))

# models and their evaluations
lr = LogisticRegression()
model_evaluation(lr)

rfc = RandomForestClassifier()
model_evaluation(rfc)

svm = SVC()
model_evaluation(svm)

dtree = DecisionTreeClassifier()
model_evaluation(dtree)

knn = KNeighborsClassifier()
model_evaluation(knn)

# Create the 'Models' directory if it does not exist
if not os.path.exists('./Phishing Site/Models'):
    os.makedirs('./Phishing Site/Models')

# Defining a function to save the model
def save_model(model, filename):
    with open(f'./Phishing Site/Models/{filename}.obj', 'wb') as f:
        pickle.dump(model, f)

# Defining a function to load the model
def load_model(filename):
    with open(f'./Phishing Site/Models/{filename}.obj', 'rb') as f:
        return pickle.load(f)

# Saving the models
save_model(rfc, 'phishingSitePredictorRfc')
save_model(dtree, 'phishingSitePredictorDtree')
save_model(knn, 'phishingSitePredictorKnn')

# Loading the models
rfcObj = load_model('phishingSitePredictorRfc')
dtreeObj = load_model('phishingSitePredictorDtree')
knnObj = load_model('phishingSitePredictorKnn')

# Evaluating loaded model
model_evaluation(rfcObj)
model_evaluation(dtreeObj)
model_evaluation(knnObj)

