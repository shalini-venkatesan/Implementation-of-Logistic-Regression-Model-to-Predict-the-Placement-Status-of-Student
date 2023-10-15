# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHALINI VENKATESAN
RegisterNumber: 212222240096
```
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

#### Placement Data

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/aa995a2a-5d06-4627-8df2-36f998247017">

#### Salary Data

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/ae80d22c-4485-465f-aa44-1a044da3d72f">

#### Checking the null function()

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/9a0709af-4dc0-40e2-aee7-85f8d1949d62">

#### Data Duplicate

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/f7e5a2f3-6547-4e2a-9b24-849d249fecd8">

#### Print Data

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/517e7058-f13f-433c-a2db-000c807f52a9">

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/e8c0a32e-c59a-4178-add3-90aacc3be152">

#### Data Status

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/733b6c40-cdbd-4650-b5e7-e37e9afc70d1">

#### y_prediction array

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/4303ec35-6123-4400-94cd-8cbc7449d8d0">

#### Accuracy value

<img width="85" alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/d67cf828-ee2b-4e22-811a-53b22553e608">

#### Confusion matrix

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/5d60dd9e-9815-4c6c-85ca-e3de5ebd2a0e">

#### Classification Report

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/c6127853-ec9f-41af-80b9-df1316dff7d0">

#### Prediction of LR

<img alt="image" src="https://github.com/JoyceBeulah/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343698/b17ec173-f112-4516-a5e8-cef746873cdb">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
