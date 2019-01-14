import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv("salary.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=1 / 3, random_state=0)

LR = LinearRegression()
LR.fit(xTrain, yTrain)
yPrediction = LR.predict(xTest)
plt.scatter(xTrain, yTrain, color='red')
plt.plot(xTrain, LR.predict(xTrain), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
