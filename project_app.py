import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
data_path = 'https://docs.google.com/spreadsheets/d/1My2DrsY_AJd9GWEe32fuh3Qzx2K-9WSu/edit?usp=drive_link&ouid=110310620693603342833&rtpof=true&sd=true'
data = pd.read_csv(data_path)

# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

data['review'] = data['review'].apply(preprocess_text)

# Convert text data to numerical representation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['review']).toarray()
y = np.where(data['sentiment'] == 'positive', 1, 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, log_reg_preds))

# Train Decision Tree model
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
dec_tree_preds = dec_tree.predict(X_test)
print("Decision Tree Report:\n", classification_report(y_test, dec_tree_preds))

# Save models and vectorizer
import pickle
pickle.dump(log_reg, open('logistic_regression_model.pkl', 'wb'))
pickle.dump(dec_tree, open('decision_tree_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("This app predicts the sentiment of a given review using Logistic Regression and Decision Tree models.")

    # Input review
    review_input = st.text_area("Enter a review:")

    if st.button("Predict Sentiment"):
        if review_input:
            # Preprocess input
            processed_input = preprocess_text(review_input)
            vectorized_input = vectorizer.transform([processed_input]).toarray()

            # Predictions
            log_reg_pred = log_reg.predict(vectorized_input)[0]
            dec_tree_pred = dec_tree.predict(vectorized_input)[0]

            # Display results
            st.write("**Logistic Regression Prediction:**", "Positive" if log_reg_pred == 1 else "Negative")
            st.write("**Decision Tree Prediction:**", "Positive" if dec_tree_pred == 1 else "Negative")
        else:
            st.write("Please enter a review for prediction.")

if __name__ == '__main__':
    main()
