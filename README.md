Text Emotion Checker
Overview

The Text Emotion Checker is a Python application designed to analyze text input and classify the emotional tone. This project leverages machine learning models to predict emotions from text, visualizes the results, and provides a user-friendly interface using Tkinter.
Features

Text Preprocessing: Cleans and preprocesses text data to improve model accuracy.
Multiple Classifiers: Utilizes Logistic Regression, Random Forest, and SVM models for emotion classification.
Model Selection: Employs Grid Search for hyperparameter tuning to find the best model.
GUI Application: Provides a graphical user interface for easy interaction and analysis.
Visualization: Displays emotion probabilities and confusion matrix using Matplotlib and Seaborn.

Installation

Clone the repository:

    

    git clone https://github.com/Wahid-najim/Text_Emotions.git

Navigate to the project directory:



    cd Text_Emotions

Install the required libraries:

bash

    pip install -r requirements.txt

   Note: Ensure that emotion.csv is present in the project directory or update the file path accordingly.

Usage

  Run the script to start the application:

  

    python text_emotion_checker.py

   Enter the text in the provided text box and click "Analyze Emotion" to see the predicted emotion and probability distribution.

Code Explanation
Text Preprocessing

The preprocess_text function cleans the text by removing punctuation and numbers, tokenizing the text, removing stop words, and stemming the tokens.
Model Training

The project uses three different classifiers (Logistic Regression, Random Forest, SVM) and selects the best one using Grid Search with cross-validation.
GUI Application

The Tkinter-based interface allows users to input text and see the predicted emotion and corresponding probabilities.
Dependencies

    pandas
    scikit-learn
    matplotlib
    seaborn
    nltk
    tkinter
