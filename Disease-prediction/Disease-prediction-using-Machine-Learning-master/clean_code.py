
import tkinter as tk
from tkinter import ttk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("E://download//elevate//Disease-prediction-using-Machine-Learning-master//Disease-prediction-using-Machine-Learning-master//Training.csv")
tr = pd.read_csv("E://download//elevate//Disease-prediction-using-Machine-Learning-master//Disease-prediction-using-Machine-Learning-master//Testing.csv")

# Define features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

X_test = tr.drop("prognosis", axis=1)
y_test = tr["prognosis"]

# Define algorithms
algorithms = {
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

# Create GUI
root = tk.Tk()
root.title("Disease Predictor")

# Create entry fields
symptoms = []
for i in range(5):
    symptom = tk.StringVar()
    symptom.set(None)
    symptoms.append(symptom)

options = sorted(X.columns)

# Create GUI widgets
name_label = tk.Label(root, text="Name of the Patient")
name_label.grid(row=0, column=0)

name_entry = tk.Entry(root)
name_entry.grid(row=0, column=1)

symptom_labels = []
symptom_menus = []
for i in range(5):
    label = tk.Label(root, text=f"Symptom {i+1}")
    label.grid(row=i+1, column=0)
    symptom_labels.append(label)

    menu = ttk.OptionMenu(root, symptoms[i], *options)
    menu.grid(row=i+1, column=1)
    symptom_menus.append(menu)

algorithm_label = tk.Label(root, text="Algorithm")
algorithm_label.grid(row=6, column=0)

algorithm = tk.StringVar()
algorithm.set(None)
algorithm_menu = ttk.OptionMenu(root, algorithm, *algorithms.keys())
algorithm_menu.grid(row=6, column=1)

def predict_disease():
    # Get selected symptoms
    selected_symptoms = [symptom.get() for symptom in symptoms]

    # Create input data
    input_data = np.zeros((1, len(X.columns)))
    for i, symptom in enumerate(selected_symptoms):
        if symptom in X.columns:
            input_data[0, X.columns.get_loc(symptom)] = 1

    # Get selected algorithm
    selected_algorithm = algorithm.get()
    if selected_algorithm in algorithms:
        model = algorithms[selected_algorithm]
    else:
        raise ValueError("Invalid algorithm selected")

    # Train model
    model.fit(X, y)

    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction
    prediction_label = tk.Label(root, text=f"Predicted disease: {prediction[0]}")
    prediction_label.grid(row=8, column=0, columnspan=2)

predict_button = tk.Button(root, text="Predict disease", command=predict_disease)
predict_button.grid(row=7, column=0, columnspan=2)

root.mainloop()