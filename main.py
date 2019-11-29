from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# =============================================================================
# Reading Irıs.csv
# =============================================================================
IrisData = pd.read_csv("iris.csv")
IrisDataNumpyArray=np.array(IrisData)
# =============================================================================
# Finding minimum, maximum, average and standard deviation values of all 
# features and printing them on the screen.
# =============================================================================
print (pd.concat([IrisData.describe()[1:4], IrisData.describe()[7:8]]))
# =============================================================================
# Plotting scatter charts for these attributes
# Sepal Length, Sepal Width
# =============================================================================
plt.subplots(3, 1, figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.scatter(IrisData['sepal_length'], IrisData['sepal_width'], c='purple',
            label='Sepal Length, Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
# =============================================================================
# Petal Length, Petal Width
# =============================================================================
plt.subplot(3, 1, 2)
plt.scatter(IrisData['petal_length'], IrisData['petal_width'], c='red',
            label='Petal Length, Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
# =============================================================================
# Sepal Length, Sepal Width and Petal Length, Petal Width
# =============================================================================
plt.subplot(3, 1, 3)
plt.scatter(IrisData['petal_length'], IrisData['petal_width'], c='red',
            label='Petal Length, Petal Width')
plt.scatter(IrisData['sepal_length'], IrisData['sepal_width'], c='purple',
            label='Sepal Length, Sepal Width')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend()
plt.savefig('Plotting.png')
# =============================================================================
# 80% of the data were randomly selected for education. 
# The remainder was used for testing.
# =============================================================================
(X_train, X_test, y_train, y_test) = train_test_split(IrisDataNumpyArray[:, 0:4],
        IrisDataNumpyArray[:, 4], test_size=0.2)
plt.subplots(1, 1, figsize=(15, 10))
# =============================================================================
# All attributes were used when creating a decision tree. 
# =============================================================================
IrisTree = tree.DecisionTreeClassifier()
# =============================================================================
# Decision tree being tested.
# =============================================================================
IrisTree = IrisTree.fit(X_train, y_train)
# =============================================================================
# Decision tree plotting.
# =============================================================================
tree.plot_tree(IrisTree)
plt.savefig('DecisionTree.png')

print ('\nConfusion Matrix')
print ((confusion_matrix(y_test, IrisTree.predict(X_test))),'\n')
print (classification_report(y_test, IrisTree.predict(X_test)))
