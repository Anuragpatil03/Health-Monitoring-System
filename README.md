# Health Monitoring System
Diabetes Prediction System is to predict whether the patients have Diabetes by developing Health Monitoring System.
Here we use Machine Learning Algorithms like KNN, SVM. Here the machine learns from the given parameters and then predicts the condition similar to true or false.

Purpose:- 
The goal of the Health Monitoring System is to Analyse the data and predict  accurate output. Motivated by this fact, Health Monitoring System aims to correctly classify input data into its underlying activity category. For this project depending on their complexity, health monitoring is focused on Diabetes Prediction. 

These modules include-

i. Dataset Collection
ii. Data Pre-processing
iii. Machine Learning  Algorithm
iv. Build Model
v. Evaluation
Let’s have a look at each model briefly.

i. Dataset Collection

This module includes data collection and understanding the data to study the patterns and trends which helps in prediction and evaluating the results. 
Dataset consists of several Medical Variables(Independent) and one Outcome Variable(Dependent)

Dataset description is given below-


• Pregnancies:-  Number of times a woman has been pregnant 

• Glucose :-  Plasma Glucose concentration of 2 hours in an oral glucose tolerance test 

• Blood Pressure :- Diastolic Blood Pressure (mm hg) 

• Skin Thickness :- Triceps skin fold thickness(mm) 

• Insulin :- 2 hour serum insulin (mu U/ml) 

• BMI :- Body Mass Index ((weight in kg/height in m)^2) 

• Age :- Age(years) 

• Diabetes Pedigree Function :- scores likelihood of diabetes based on family history)

 • Outcome :- 0 (doesn't have diabetes) or 1 (has diabetes)




ii. Data Pre-processing

PIDD-Pima Indians Diabetes Dataset
The proposed methodology is evaluated on Diabetes Dataset namely (PIDD), which is taken from UCI Repository. This dataset comprises of medical detail of 768 instances which are female patients. The dataset also comprises
numeric-valued 8 attributes where value of one class ’0’ treated as tested negative for diabetes and value of another class ’1’ is treated as tested positive for diabetes.
This phase of model handles inconsistent data in order to get more accurate and precise results. This dataset contains missing values. So we imputed missing values for few selected attributes like Glucose level, Blood Pressure, Skin Thickness, BMI and Age because these attributes cannot have values zero. Then we scale the dataset to normalize all values.


iii. Machine Learning Algorithm

 k-Nearest Neighbours: 
The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training data set. To make a prediction for a new data point, the algorithm finds the closest data 
points in the training data set, its “nearest neighbours.”
First, let’s investigate whether we can confirm the connection between model complexity and accuracy:
The above plot shows the training and test set accuracy on the y-axis against the setting of n_ neighbours on the x-axis. Considering if we choose one single nearest neighbour, the prediction on the training set is perfect. But 
when more neighbours are considered, the training accuracy drops, indicating that using the single nearest 
neighbour leads to a model that is too complex. The best performance is somewhere around 9 neighbours



iv. Model Building
This is most important phase which includes model building for prediction of diabetes. In this we have implemented Decision Tree machine learning algorithms for diabetes prediction. These algorithms include 
flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
*********************************************************************************************************************************************************

CONCLUSION**

In this study, KNN &SVM machine learning algorithms are applied on the dataset and the classification has been done using algorithms of which Support Vector Machine gives highest accuracy of 80% for PIMS Diabetes Dataset.
We have seen comparison of machine learning algorithm
accuracies with datasets. It is clear that the model improves accuracy and precision of diabetes prediction with this dataset compared to existing dataset. Further this work can be extended to find how likely non-diabetic people can have diabetes in next few years





