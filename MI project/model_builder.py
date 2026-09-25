# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading data...")
data = pd.read_csv('dataset\\train.csv')
X = data.drop('label', axis=1)
y = data['label']

print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X, y)

print("Saving model...")
with open('model\\model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print(" Model saved as model.pkl")
