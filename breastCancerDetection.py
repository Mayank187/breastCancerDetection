# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC

dataset = pd.read_csv('breastCancer.csv')


# Data preprocessing phase
data_bare_nucleoli = pd.DataFrame(dataset.bare_nucleoli)

#Replacing all Non integer value with NAN value
for i in dataset.bare_nucleoli.values:
    if str(i).isdigit():
        continue
    else:
        dataset.bare_nucleoli = dataset.bare_nucleoli.replace(i, np.nan)

#Filling all NAN values with the median value
dataset.bare_nucleoli = dataset.bare_nucleoli.fillna(int(dataset.bare_nucleoli.median()))

#Changing the data type to int64 for bare_nucleoli field
dataset.bare_nucleoli = dataset.bare_nucleoli.astype('int64')

#Exploratory Data Analysis
#First column is ID which is not needed for prediction

#Dropping ID Column
dataset.drop('id',axis=1,inplace=True)
print(dataset.describe().T)

#Bivariate Data Analysis
sns.displot(dataset['class'])
plt.savefig('Displot.jpg')

#Multivariate Data Analysis
fig = dataset.hist(bins=20,figsize=(30,30),layout=(6,3))
plt.savefig('Histogram.jpg')

#Box Plot
plt.figure(figsize=(70,30))
sns.boxplot(data=dataset,orient='h')
plt.savefig('BoxPlot.jpg')

#Correlation
plt.figure(figsize=(70,30))
sns.heatmap(dataset.corr(),vmax=1,annot=True,square=True,cmap='viridis')
plt.title('Correlation between different attributes')
plt.savefig('Heatmap.jpg')
plt.show()


#Pair Plot
sns.pairplot(dataset, diag_kind='kde')
plt.savefig('PairPlot.jpg')


#Building the model
X = dataset.drop('class',axis=1)
Y = dataset['class']

#Splitting Data set in 70:30
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

#K Nearest Classifier
knnClassifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
knnClassifier.fit(X_train,Y_train)
Y_pred = knnClassifier.predict(X_test)

#Classification Report
cr = classification_report(Y_test,Y_pred)
print(cr)

#Confusion matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
plt.figure(figsize=(35,15))
sns.heatmap(cm,annot=True)
plt.savefig('KNN_Confusion_Matrix.jpg')

#Support Vector Machine
svc = SVC(gamma=0.025,C=3)
svc.fit(X_train,Y_train)
y_svc_pred = svc.predict(X_test)

#Classification Report
cr_svc = classification_report(Y_test,y_svc_pred)
print(cr_svc)


#Confusion matrix
cm_svc = confusion_matrix(Y_test,y_svc_pred)
print(cm_svc)
plt.figure(figsize=(35,15))
sns.heatmap(cm_svc,annot=True)
plt.savefig('SVC_Confusion_Matrix.jpg')
