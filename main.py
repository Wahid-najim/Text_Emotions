# Importing all necessarry library we need for this project 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, Label, Button, Text
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources (if not already done)
nltk.download('punkt')
nltk.download('stopwords')

# Function for text preprocessing
def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize and remove stop words
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load the data
data = pd.read_csv('emotion.csv')

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)
X = data['text']
y = data['emotion']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with TF-IDF and different classifiers
pipelines = {
    'logistic_regression': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'random_forest': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100))
    ]),
    'svc': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', SVC(probability=True))
    ])
}

# Hyperparameters for GridSearchCV
param_grids = {
    'logistic_regression': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__penalty': ['l1', 'l2']
    },
    'random_forest': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__n_estimators': [50, 100, 200]
    },
    'svc': {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf']
    }
}

# Grid search and model selection
best_score = 0
best_model = None
best_pipeline_name = ''

for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Compare models
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_pipeline_name = name

print(f"Best model: {best_pipeline_name}")

# Evaluate the best model
y_pred = best_model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix:")
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Create a function to predict emotion and plot results
def analyze_text():
    input_text = text_entry.get("1.0", "end-1c")
    if input_text:
        input_text = preprocess_text(input_text)
        prediction = best_model.predict([input_text])[0]
        prediction_proba = best_model.predict_proba([input_text])[0]
        emotions = best_model.classes_

        # Show result
        result_label.config(text=f"Predicted Emotion: {prediction}")

        # Plot results
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotions, y=prediction_proba, palette='coolwarm')
        plt.title("Emotion Probability Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        # Adding percentage labels
        for i, value in enumerate(prediction_proba):
            plt.text(i, value + 0.02, f"{value:.2%}", ha='center', color='black', fontsize=12, weight='bold')

        plt.show()

# Create the GUI
root = Tk()
root.title("Emotion Recognition")

Label(root, text="Enter Text:").pack(pady=10)

text_entry = Text(root, height=5, width=50)
text_entry.pack(pady=10)

analyze_button = Button(root, text="Analyze Emotion", command=analyze_text)
analyze_button.pack(pady=10)

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
