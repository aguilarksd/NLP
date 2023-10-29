# NLP using Bag of Words (BoW) for Sentiment Analysis

## Project Description

This project demonstrates the use of Natural Language Processing (NLP) techniques to classify restaurant reviews as positive or negative using the Bag of Words (BoW) model. The goal is to create a machine-learning model that can automatically determine the sentiment of a given review text.

## Dataset

The dataset used for this project is in the form of a TSV (Tab-Separated Values) file named "Restaurant_Reviews.tsv." It contains 1000 restaurant reviews, each labeled as either positive or negative. The "Review" column contains the text of the reviews, and the "Liked" column indicates the sentiment (1 for positive, 0 for negative).

## Preprocessing

The project includes data preprocessing steps to prepare the text data for model training:

- Text Cleaning: The text is cleaned by removing non-alphabetical characters and converting all text to lowercase.
- Tokenization: The text is split into individual words to create a list of words in each review.
- Stopword Removal: Common stopwords are removed to improve the quality of the text data.
- Stemming: Words are reduced to their root form using the Porter stemming algorithm.

## Bag of Words (BoW) Model

The Bag of Words model is created using the scikit-learn library. It involves the following steps:

- Text Vectorization: The text data is transformed into a numerical format using the CountVectorizer from scikit-learn. It creates a histogram of the most frequently occurring 1500 words.
- Feature Matrix: The resulting feature matrix represents each review as a vector of word counts.
- Target Variable: The target variable is extracted from the dataset, indicating whether the review is positive or negative (binary classification).

## Classification Models

Several classification models are applied to the data, and their performance is evaluated using various metrics:

1. Gaussian Naive Bayes (G-NB)
2. K-Nearest Neighbors (K-NN)
3. Logistic Regression (LR)
4. Random Forest Classification (RF)
5. Decision Tree Classification (DT)
6. Support Vector Machine (SVM)
7. CART Model
8. Maximum Entropy Model

Each model is trained, and predictions are made on a test set. The following metrics are calculated for each model:

- Accuracy
- Precision
- Recall
- F1 Score

## Model Comparison

The project provides a visual comparison of the different classification techniques using a bar chart. The chart displays the metrics (accuracy, precision, recall, and F1 score) for each technique, allowing for a quick assessment of model performance.

## Best Results

The project also identifies the best-performing model for each metric (Accuracy, Precision, Recall, and F1 Score). The best techniques for each metric are printed along with their corresponding values.

### Best Accuracy: SVM - 0.78
### Best Precision: RF - 0.96
### Best Recall: G-NB - 0.82
### Best F1 Score: SVM - 0.80

## Conclusion

This project demonstrates the effectiveness of NLP and BoW in sentiment analysis of restaurant reviews. It provides insights into the performance of various classification models and helps identify the best-performing techniques for the task.

Feel free to use this code and README as a starting point for your project. Make sure to replace 'Restaurant_Reviews.tsv' with your own dataset if necessary and adapt the code to your specific needs.
