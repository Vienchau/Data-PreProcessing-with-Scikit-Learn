import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_df = pd.read_csv("./data.csv")
print(data_df.head())
print(type(data_df))

#columns information
print(data_df.info())

#missing percent in each column
for col in data_df.columns:
	missing_data = data_df[col].isna().sum()
	missing_percent = missing_data/len(data_df)*100
	print(f"column: {col} has {missing_percent}%")

#missing data with figure

fig, ax= plt.subplots(figsize=(5,5))
sns.heatmap(data_df.isna(), cmap="Blues",linewidth=2, cbar = False)

#split data
X = data_df.iloc[:, :-1].values
y = data_df.iloc[:, -1].values

print(X)
print(y)


#process missing data with SimpleImputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#Encoding Independent varialbe X

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = "passthrough")
X = ct.fit_transform(X)
print(X)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

#data split to test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
print("X train after standardisation:")
print(X_train)
print("X_test after standardisation")
print(X_test)



plt.show()
