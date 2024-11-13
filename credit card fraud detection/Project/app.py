from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

app = Flask(__name__)

# Route to display upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        
        # Process the CSV and train the model
        data = pd.read_csv(file_path)

        # Split into features and labels
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

        # Save metrics to CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('training_metrics.csv', index=False)

        # Plot accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        accuracy_plot_path = os.path.join('static', 'accuracy_plot.png')
        plt.savefig(accuracy_plot_path)
        plt.close()

        # Plot loss
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        loss_plot_path = os.path.join('static', 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()

        return redirect(url_for('show_results'))

# Route to display the results after training
@app.route('/results')
def show_results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
