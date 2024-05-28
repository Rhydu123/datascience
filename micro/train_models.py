# Importing essential libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Load dataset
df = pd.read_csv('heart.csv')

# Preprocess the data
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
standScaler = StandardScaler()
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])

# Splitting the dataset into features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Train KNeighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=12)
knn_classifier.fit(X, y)
joblib.dump(knn_classifier, 'knn_model.joblib')

# Train Decision Tree Classifier
decision_classifier = DecisionTreeClassifier(max_depth=3)
decision_classifier.fit(X, y)
joblib.dump(decision_classifier, 'decision_tree_model.joblib')

# Train Random Forest Classifier
forest_classifier = RandomForestClassifier(n_estimators=90)
forest_classifier.fit(X, y)
joblib.dump(forest_classifier, 'random_forest_model.joblib')
