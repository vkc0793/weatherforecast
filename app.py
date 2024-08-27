from flask import Flask, render_template, request
import sqlite3
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example usage

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/contact", methods=["GET", "POST"])
def contactus():
    if request.method == "POST":
        fname = request.form.get("fullname")
        pno = request.form.get("phone")
        email = request.form.get("email")
        addr = request.form.get("address")
        msg = request.form.get("message")
        conn = sqlite3.connect("weather.db")
        curr = conn.cursor()
        curr.execute(f'''INSERT INTO CONTACT VALUES("{fname}", "{pno}", "{email}", "{addr}", "{msg}")''')
        conn.commit()
        conn.close()
        return render_template("message.html")
    else:
        return render_template('contact.html')

@app.route("/forecast")
def analytical():
    return render_template("forecast.html")

@app.route('/predictor', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        time_steps = int(request.form['time_steps'])
        start_date=datetime.now().date()

        # Load the CSV file
        X_combined = pd.read_csv('X_combined.csv')

        # Normalize numerical data
        scaler = MinMaxScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)

        # Load the label_encoders dictionary
        with open('label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)

        # Load the LSTM model
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Get the most recent time_steps of data as the initial input for forecasting
        last_sequence = X_combined_scaled[-time_steps:]  # Shape: (time_steps, num_features)

        # Initialize a list to store predictions
        future_predictions = []

        # Perform iterative forecasting for the next 10 days
        for _ in range(time_steps):  # Predicting for the next 10 days
            # Reshape the input to match the LSTM input shape
            input_data = last_sequence.reshape((1, time_steps, last_sequence.shape[1]))

            # Make a prediction
            next_day_prediction = model.predict(input_data)

            # Inverse transform the prediction to get it back to the original scale
            next_day_prediction = scaler.inverse_transform(next_day_prediction)

            # Append the prediction to the list
            future_predictions.append(next_day_prediction[0])

            # Update the sequence: remove the oldest day and add the new prediction
            last_sequence = np.vstack((last_sequence[1:], next_day_prediction))

        # Convert predictions to a DataFrame for better readability
        future_predictions_df = pd.DataFrame(future_predictions, columns=X_combined.columns)

        # Handle categorical predictions
        X_cat = ['preciptype', 'uvindex', 'severerisk', 'conditions', 'description']
        X_num = 21
        # Handling categorical predictions as before
        for col_idx, col_name in enumerate(X_cat, start=X_num):
            predicted_cat = future_predictions_df.iloc[:, col_idx].values
    
            # Round and clip to ensure values fall within the valid label range
            predicted_cat_rounded = np.round(predicted_cat).astype(int)
            predicted_cat_rounded = np.clip(predicted_cat_rounded, 0, len(label_encoders[col_name].classes_) - 1)
    
            # Convert back to original categorical labels
            future_predictions_df[col_name] = label_encoders[col_name].inverse_transform(predicted_cat_rounded)

        # Generate the dates for the forecast period
        future_dates = [start_date + timedelta(days=i) for i in range(time_steps)]
        future_predictions_df.insert(0, 'date', future_dates)

        # Drop unwanted columns
        columns_to_drop = ['feelslikemax', 'feelslikemin','dew','precipprob','precipcover','windgust','	winddir','sealevelpressure','cloudcover','solarradiation','	solarenergy','moonphase','	daylight_hours','uvindex','severerisk','feelslike','winddir','solarenergy','daylight_hours','precip']  # Specify the columns to drop
        future_predictions_df = future_predictions_df.drop(columns=columns_to_drop, errors='ignore')

        # Convert predictions to HTML table
        predictions_html = future_predictions_df.to_html(index=False)

        # Save the DataFrame as an HTML file
        future_predictions_df.to_html('predictions.html', index=False)

        # Render the HTML table in the predictor.html template
        return render_template('results.html', table=predictions_html)
    else:
        return render_template('predictor.html')

if __name__ == "__main__":
    app.run(debug=True)
