import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the training dataset
train_data_path = "C:\\Users\\HANIF\\Downloads\\fake-news\\train.csv"
train_data = pd.read_csv(train_data_path)

# Preprocessing
train_data = train_data.dropna()  # Remove missing values
X_train = train_data['text']  # Feature
y_train = train_data['label']  # Target

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit the vectorizer on the training data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_val_tfidf = vectorizer.transform(X_val_split)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train_split)

# Predict on the validation set
y_val_pred = model.predict(X_val_tfidf)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val_split, y_val_pred)
val_precision = precision_score(y_val_split, y_val_pred)
val_recall = recall_score(y_val_split, y_val_pred)
val_f1 = f1_score(y_val_split, y_val_pred)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Precision: {val_precision}')
print(f'Validation Recall: {val_recall}')
print(f'Validation F1 Score: {val_f1}')

# Save the model and fitted vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# Load the test dataset
test_data_path = "C:\\Users\\HANIF\\Downloads\\fake-news\\test.csv"
test_data = pd.read_csv(test_data_path)

# Preprocess the test data
test_data = test_data.dropna(subset=['text'])  # Remove rows where 'text' is missing
X_test = test_data['text']

# Transform the test data using the fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Predict on the test data
y_test_pred = model.predict(X_test_tfidf)

# Optionally, save the test predictions
test_data['predictions'] = y_test_pred
test_data.to_csv("C:\\Users\\HANIF\\Downloads\\fake-news\\test_predictions.csv", index=False)
