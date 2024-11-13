from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, Dropout

app = Flask(__name__)

# Function to load model and make predictions
def train_and_predict(data_path):
    data = pd.read_csv(data_path)
    
    # Data Preprocessing
    non_fraud = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    
    # Balance the data
    non_fraud = non_fraud.sample(fraud.shape[0])
    data = pd.concat([fraud, non_fraud], ignore_index=True)
    
    # Split the data into features and labels
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for Conv1D input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Model definition
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=X_train[0].shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model (reduced epochs for faster testing)
    model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))
    
    # Make predictions
    pred = model.predict(X_test)
    
    # Create a DataFrame for results
    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': pred.flatten()
    })
    
    return results

# Route to upload the file and display results
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file temporarily
            file_path = "./" + file.filename
            file.save(file_path)
            
            # Run the prediction function
            results = train_and_predict(file_path)
            
            # Render the results on the result.html page
            return render_template("result.html", tables=[results.to_html(classes='data')], titles=results.columns.values)
    
    # Render the main page with file upload option
    return render_template("index.html")

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
