import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the models
knn_model = joblib.load('knn_model.joblib')
decision_tree_model = joblib.load('decision_tree_model.joblib')
random_forest_model = joblib.load('random_forest_model.joblib')

# Create the GUI window
window = tk.Tk()
window.title("Heart Disease Prediction")
window.geometry("400x400")

# Input fields for the features
entries = {}
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for feature in features:
    tk.Label(window, text=feature).pack()
    entry = tk.Entry(window)
    entry.pack()
    entries[feature] = entry

# Dropdown menu for model selection
model_var = tk.StringVar(window)
model_var.set("KNN")  # default value
tk.Label(window, text="Select Model").pack()
tk.OptionMenu(window, model_var, "KNN", "Decision Tree", "Random Forest").pack()

# Function to get the input values and predict
def predict():
    input_data = np.array([[float(entries[feature].get()) for feature in features]])
    model_name = model_var.get()
    if model_name == "KNN":
        model = knn_model
    elif model_name == "Decision Tree":
        model = decision_tree_model
    else:
        model = random_forest_model
    prediction = model.predict(input_data)[0]
    messagebox.showinfo("Prediction", f"The predicted target is: {prediction}")

# Predict button
tk.Button(window, text="Predict", command=predict).pack()

# Run the GUI loop
window.mainloop()
