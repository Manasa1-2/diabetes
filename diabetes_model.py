# diabetes_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dataset = pd.read_csv(url, header=None, names=column_names)

X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:, 8].values

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

loss, accuracy = model.evaluate(X, Y)
print(f'Accuracy: {accuracy*100:.2f}%')

predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype("int32")

for i in range(10):
    print(f'Predicted: {predicted_classes[i][0]}, True: {Y[i]}')
