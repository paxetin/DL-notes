from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names:" + str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=iris['target'])
iris_data = pd.concat([x, y], axis=1)
iris_data = iris[['sepal length (cm)', 'petal length (cm)', 'target']]
iris_data = iris_data[iris_data['target'].isin([0, 1])]
iris_data.head(3)


x_train, x_test, y_train, y_test = train_test_split(
    iris_data[['sepal length (cm)', 'petal length (cm)']], iris_data[['target']], test_size=0.3, random_state=0
)


sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


svm = SVC(kernel='linear', probability=True)
svm.fit(x_train_std, y_train['target'].values)

svm.predict(x_test_std)
y_test['target'].values

error = 0
for i, v in enumerate(svm.predict(x_test_std)):
    if v != y_test['target'].values[i]:
        error+1
print(error)
