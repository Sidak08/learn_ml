# coloab link https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=D4kPWqBYVDlj

#qestions

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print(dftrain.describe())
print(dftrain.shape)
print(y_train.head())

# Plotting commands
dftrain.age.hist(bins=20)
# plt.show()

dftrain.sex.value_counts().plot(kind='barh')
# plt.show()

dftrain['class'].value_counts().plot(kind='barh')
# plt.show()

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

# Age vs. Fare Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(dftrain['age'], dftrain['fare'], alpha=0.5)
plt.title('Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Survival Rate by Passenger Class
plt.figure(figsize=(10, 6))
pd.concat([dftrain, y_train], axis=1).groupby('class').survived.mean().plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Distribution of Fare by Survival Status
plt.figure(figsize=(10, 6))
dftrain['fare'].hist(by=y_train, bins=20, alpha=0.7)
plt.suptitle('Distribution of Fare by Survival Status')
plt.show()

# Survival Rate by Embarkation Port
plt.figure(figsize=(10, 6))
pd.concat([dftrain, y_train], axis=1).groupby('embark_town').survived.mean().plot(kind='bar')
plt.title('Survival Rate by Embarkation Port')
plt.xlabel('Embarkation Port')
plt.ylabel('Survival Rate')
plt.show()

# Age Distribution by Survival Status
plt.figure(figsize=(10, 6))
dftrain['age'].hist(by=y_train, bins=20, alpha=0.7)
plt.suptitle('Age Distribution by Survival Status')
plt.show()
