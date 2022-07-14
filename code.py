import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

diab = pd.read_csv('/content/diabetes.csv') // import the Pima Indians Dataset available on kaggle
diab.head()
diab.shape
diab.info()
diab.duplicated().sum()
diab.describe().T
diab.corr().style.background_gradient(cmap='coolwarm')
diab.columns
sns.scatterplot(data=diab, x=diab['Pregnancies'], y=diab['BloodPressure'], hue=diab['Outcome'])
sns.scatterplot(data=diab, x=diab['SkinThickness'], y=diab['BMI'], hue=diab['Outcome'])
sns.scatterplot(data=diab, x=diab['Glucose'], y=diab['Insulin'], hue=diab['Outcome'])
diab['Outcome'].value_counts()
x = diab.drop('Outcome',axis=1)
y = diab['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x,y) 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=17, metric='manhattan')
knn.fit(X_train_std, y_train)
y_pred_knn = knn.predict(X_test_std)
y_pred_knn
knn_cm = confusion_matrix(y_test, y_pred_knn)
print(knn_cm)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(knn_accuracy)
print("Accuracy on training set: {:.3f}".format(knn.score(X_train_std, y_train)))
print("Accuracy on test set: {:.3f}".format(knn.score(X_test_std, y_test)))
# this loop will tell us which k value is giving us best accuracy.
scores = []
for k in range(1,30):
    knn = KNeighborsClassifier(k).fit(x, y)
    scores.append(knn.score(x, y))

print(scores, end = " ")
x_1 = diab.drop('Outcome',axis=1)
y_1 = diab['Outcome']
x_1.shape, y_1.shape
# dividing our data
X_train, X_test, y_train, y_test = train_test_split(x_1,y_1) 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=0.9) # No hyperparameter given means it will use default values (i.e. - C = 1.0 and kernel = 'rbf')
svc.fit(X_train_std, y_train)
y_pred_svc = svc.predict(X_test_std)
y_pred_svc
svc = SVC(C=0.9) # No hyperparameter given means it will use default values (i.e. - C = 1.0 and kernel = 'rbf')
svc.fit(X_train_std, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_std, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_std, y_test)))
algorithms=['KNN','SVM']
scores=[knn_accuracy,svc_accuracy]
# sns.set(rc={'figure.figsize':(10,5)})
# sns.l(x=algorithms,y=scores)
plt.bar(algorithms, scores)

***************************************************************************************************************************************************8
