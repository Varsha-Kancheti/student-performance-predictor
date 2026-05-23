import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Features (input)
X = data[['study_hours', 'attendance', 'sleep_hours']]

# Target (output)
y = data['score']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
sleep_hours = float(input("Enter sleep hours: "))

# Create dataframe for prediction
input_data = pd.DataFrame(
    [[study_hours, attendance, sleep_hours]],
    columns=['study_hours', 'attendance', 'sleep_hours']
)

# Predict score
prediction = model.predict(input_data)

# Output result
print("Predicted Score:", round(prediction[0], 2))