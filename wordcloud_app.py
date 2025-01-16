import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    try:
        text = re.sub(r'\W+', ' ', text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return ""

def main():
    st.title("PDF Text Processor and WordCloud Generator")
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file is not None:
        st.write("Processing the uploaded PDF...")
        raw_text = extract_text_from_pdf(pdf_file)
        st.write("### Extracted Text:")
        st.text_area("Extracted Text", raw_text[:1000], height=200)

        if raw_text.strip():
            processed_text = preprocess_text(raw_text)
            st.write("### Preprocessed Text:")
            st.text_area("Preprocessed Text", processed_text[:1000], height=200)

            if st.button("Generate WordCloud"):
                wordcloud = WordCloud(width=1600, height=800, max_words=50000, background_color="white").generate(processed_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
        else:
            st.warning("No text extracted from the uploaded PDF. Please upload a valid PDF file.")

if __name__ == "__main__":
    main()
