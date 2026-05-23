from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['study_hours', 'attendance', 'sleep_hours']]
y = data['score']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        sleep_hours = float(request.form['sleep_hours'])

        input_data = pd.DataFrame(
            [[study_hours, attendance, sleep_hours]],
            columns=['study_hours', 'attendance', 'sleep_hours']
        )

        prediction = round(model.predict(input_data)[0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)