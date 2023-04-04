import matplotlib.pyplot as plt
import seaborn as sns

sns.get_dataset_names()
df = sns.load_dataset("titanic")
sns.countplot(x="who", hue="survived", data=df)
plt.show()

sns.histplot(x='age', kde=True, bins=5, hue=df['survived'], data=df)
plt.show()