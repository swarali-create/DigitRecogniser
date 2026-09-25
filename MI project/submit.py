import pandas as pd
import pickle

# Load model
with open('model\\model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test.csv
test = pd.read_csv('dataset\\test.csv')

# Predict labels
predictions = model.predict(test)

# Save results
output = pd.DataFrame({'ImageId': range(1, len(predictions)+1), 'Label': predictions})
output.to_csv('dataset\\submission.csv', index=False)

print("✅ Predictions saved to submission.csv")
