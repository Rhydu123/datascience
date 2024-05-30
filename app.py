from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('heart.csv')

# Data Preprocessing
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])
X = dataset.drop('target', axis=1)
y = dataset['target']

# Model selection
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X, y)

random_forest = RandomForestClassifier(n_estimators=90)
random_forest.fit(X, y)

models = {
    'knn': knn,
    'decision-tree': decision_tree,
    'random-forest': random_forest
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"})
    
    model_name = data.get('model')
    input_data = data.get('input')
    
    if not model_name or not input_data:
        return jsonify({"error": "Invalid input"})
    
    if model_name not in models:
        return jsonify({"error": "Invalid model selected"})
    
    model = models[model_name]
    
    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    input_df = input_df[X.columns]
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
    
    prediction = model.predict(input_df)
    return jsonify({"prediction": int(prediction[0])})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    file = request.files.get('file')
    model_name = request.form.get('model')
    
    if not file or not model_name:
        return jsonify({"error": "No file uploaded or model selected"})
    
    # Load the dataset
    df = pd.read_csv(file)

    # Data Preprocessing
    dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])
    X = dataset.drop('target', axis=1)
    y = dataset['target']

    # Model selection
    if model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=12)
    elif model_name == 'decision-tree':
        model = DecisionTreeClassifier(max_depth=3)
    elif model_name == 'random-forest':
        model = RandomForestClassifier(n_estimators=90)
    else:
        return jsonify({"error": "Invalid model selected"})
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=10)
    accuracy = round(scores.mean(), 4) * 100

    return jsonify({"accuracy": f"{accuracy}%"})

if __name__ == '__main__':
    app.run(debug=True)
