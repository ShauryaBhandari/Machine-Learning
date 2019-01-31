import numpy as np
# Assigning features and label variables
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
           'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# Import LabelEncoder
from sklearn import preprocessing
# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
print (weather_encoded)

# Converting string labels into numbers
temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)
print ("\nTemp:", temp_encoded)
print ("Play:", label)

# Combinig weather and temp into single listof tuples
features = zip(weather_encoded, temp_encoded)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()


# Train the model using the training sets
model.fit(features, label)

# Predict Output
predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)
