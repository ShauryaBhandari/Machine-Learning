import pandas as pd

dataset = pd.read_csv("bill_authentication.csv")
print("Shape of the data is:", dataset.shape)  # Data analysis
print("First five data entries are:\n", dataset.head())  # Check first five entries

X = dataset.drop("Class", axis=1)  # First four columns
y = dataset["Class"]  # Last output class

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#print(y_pred) # Remove comment to see predictions

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))
