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
    # WordCloud customization
    st.write("### WordCloud Customization")
    
    col1, col2,col3 = st.columns(3)
    with col1:
        bg_color = st.color_picker("WordCloud Background", "#000000")
    with col2:
        width = st.slider("Width (in pixels):", min_value=100, max_value=1920, value=1200)
    with col3:
        height = st.slider("Height (in pixels):", min_value=100, max_value=1080, value=600)
    

    if user_text:
        with st.spinner("Preprocessing text..."):
            processed_text = preprocess_text(user_text)
        if st.button("Generate WordCloud"):       
            # Generate WordCloud
            with st.spinner("Generating WordCloud..."):
                wordcloud = WordCloud(width=width, height=height, max_words=1000, background_color=bg_color).generate(processed_text)
                plt.figure(figsize=(width/100, height/100))  # Convert pixels to inches for Matplotlib
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)

if __name__ == "__main__":
    main()
