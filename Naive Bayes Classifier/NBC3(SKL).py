from sklearn import datasets
wine = datasets.load_wine()
print("Features:", wine.feature_names)
print("\nLabels:", wine.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)  # 70% training and 30% test

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("\n")
print(y_pred)
from sklearn import metrics
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
