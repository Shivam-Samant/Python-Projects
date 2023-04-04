import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./CSV/income.csv')
print(df.head())
print(df.shape)

# scatter plot -> age vs income
# plt.scatter(df.Age, df['Income($)'], color="r", marker='+')
# plt.xlabel('Age')
# plt.ylabel('Income')
# plt.title("Age vs Income")
# plt.show()


SSE = []

for i in range(1, 10): 
    # init KMeans model
    kmeans = KMeans(n_clusters=i)

    # fitting cluster
    y_pred = kmeans.fit_predict(df[['Age', 'Income($)']])
    # print(y_pred)

    # concat cluster data
    df_y_pred = pd.Series(y_pred, name="Cluster")
    df1 = pd.concat([df, df_y_pred], axis=1)
    # print(df1.head())

    # kmeans cluster centroid
    # print(kmeans.cluster_centers_)

    # intertia -> tell mean squared error
    intertia = kmeans.inertia_
    SSE.append(intertia)


print(SSE)

# plotting K vs SSE graph
plt.plot(np.arange(1,10), SSE)
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("K vs SSE")
plt.show()







# CSIR AICR

# gamma high  value far
# gamma low   value near