Prediction using Supervised ML ( Prediction of Marks )
 
We are going to discuss and implement knowledge in the following areas:
What is Machine learning?
What is Supervised and Unsupervised ML?
What is Scikit Learn?
Practical Implementation to predict marks of a student based on the number of hours studied
What is Machine Learning?
 
Machine Learning is a subset of Artificial Intelligence, it may be defined as the process or study of computer algorithms in such a way that it helps to predict a potential decision or optimum decision based on data provided and scenarios.
 
It is one of the exciting fields in Computer Science that enables the practitioner to improve the efficiency of the accuracy percentage by optimizing algorithms.
 
What is Supervised and Unsupervised ML?
 
Supervised Machine Learning is a method where the models are trained using labeled data, it needs supervision to train the model.
 
In unsupervised Machine Learning extraction of features and patterns takes place as it includes unlabeled data in the scenario.
 
What is Scikit Learn? 
 
The Scikit Learn is written in Python. The Scikit Learn uses Numpy for the operation of high-performance linear array and algebra.
 
Prediction of marks of a student based on the number of hours he/she studies 
 
The first step is to import the libraries that are required for the implementation of the code operation.  Here we import the Pandas to import and analyze data, NumPy to perform the multi-dimensional operation, and matplotlib to perform graphical plot into the context.
import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt    
The next phase is to load the data into the program to perform the desired operation. Here we use the pandas to load the excel data and when data is successfully loaded we print a statement  to get confirmation.

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
  
print("Successfully imported data into console" )  

Next is to view the data and so we are using the head() function. The head function default view is the top five, but again whatever you want to be in the number of views you can do it as entered, in this case, I have entered six views.

data_load.head(10)  

The next phase is to enter distribution scores and plot them according to the requirement, here we are going to enter the title, x_label, and y_label, and show it according to the desired result.

data_load.plot(x='Hours', y='Score', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show()  

The process of dividing the data into attributes and labels is our next task, so we implement the same as below.

X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values  

The split of data into the training and test sets is very important as in this time we will be using Scikit Learn's builtin method of train_test_split(), as below:
from sklearn.model_selection import train_test_split    

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)   
The very next process is to train the algorithm, thus the step include the following:

from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training ... Completed !.")  

The very next phase is to implement the plotting test data using the previously trained test data:
line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()  

Predicting the scores for the model is the next important step towards knowing our model, as we proceed as follows,
print(X_test)   

y_pred = regressor.predict(X_test)  

Comparing the actual versus predicted model to understand our model fitting,

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df 

Now it's time to test our model with sample testing hours, so in this case, we take 7 hours, i.e, if a student studies for seven hours, approximately how many marks he can get based on the data we received and the model we applied.

hours = [[7]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0]))  

Now we have successfully implemented the model and have received the output, the important thing that needs to be kept in mind  is that this model works only for the dataset we provided, the results may change if the data is changed and thus we need to optimize the model again.
