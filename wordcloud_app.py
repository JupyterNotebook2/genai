import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Function to preprocess text
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Streamlit app
def main():
    st.title("Text Preprocessor and WordCloud Generator")

    # User input text
    user_text = st.text_area("Enter your text:", height=200)
    if user_text:
        # Preprocess text
        with st.spinner("Preprocessing text..."):
            processed_text = preprocess_text(user_text)
        st.write("### Preprocessed Text:")
        st.text_area("Processed Text", processed_text, height=150)

        # WordCloud customization
        st.write("### WordCloud Customization")
        bg_color = st.color_picker("Choose a background color for the WordCloud", "#FFFFFF")

        # Generate WordCloud
        if st.button("Generate WordCloud"):
            with st.spinner("Generating WordCloud..."):
                wordcloud = WordCloud(width=800, height=400, max_words=500, background_color=bg_color).generate(processed_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)

if __name__ == "__main__":
    main()
