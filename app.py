# app.py
import streamlit as st
from rag import RAGService
from summarization import generate_summary
from preprocessing import extract_text_from_pdf
import io
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to generate a unique key based on file content
def generate_file_key(uploaded_file):
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_contents).hexdigest()
            return file_hash
        except Exception as e:
            logging.error(f"Error generating file key: {e}")
            st.error("Error processing uploaded file.  Please check the logs.")
            return None  # Return None on error
    return None

# Function to handle file upload and processing, NOW CACHED
@st.cache_data()
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with st.spinner("Processing the PDF..."):
            try:
                bytes_data = uploaded_file.getvalue()
                text = extract_text_from_pdf(io.BytesIO(bytes_data))
                if text:
                    return text
                else:
                    st.error("Error extracting text from PDF. Please check the logs.")
                    return None
            except Exception as e:
                logging.error(f"Error processing uploaded file: {e}")
                st.error("Error processing uploaded file.  Please check the logs.")
                return None  # Return None on error
    return None

# --- App UI ---
st.title("PDF Question Answering with RAG")
st.write("Upload a PDF and ask questions about it.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
file_key = generate_file_key(uploaded_file)  # Get the unique file key
text = process_uploaded_file(uploaded_file)  # Get extracted text

# Use st.cache_resource with a key based on the file.
@st.cache_resource(show_spinner=False)
def get_rag_service(key):  # Only the KEY is needed now
    if key is not None:
        try:
            text = process_uploaded_file(uploaded_file)
            if text:
              return RAGService(text) # Create a NEW RAGService here
            return None
        except Exception as e:
            logging.error(f"Error initializing RAGService: {e}")
            st.error("Error initializing RAG service. Please check the logs and try again.")
            return None
    return None

if file_key:
    rag_service = get_rag_service(key=file_key)

    if rag_service:
        with st.expander("Show Summary"):
            summary = generate_summary(rag_service.text)
            if summary:
              st.write(summary)
            else:
              st.error("Error in generating summary")

        user_query = st.text_input("Enter your question:")

        if user_query:  # This is true if the string is NOT empty
            with st.spinner("Thinking..."):
                try:
                    answer = rag_service.generate_answer(user_query)
                    st.write(answer)
                except Exception as e:
                    logging.error(f"Error generating answer: {e}")
                    st.error("An error occurred while generating the answer.")
        elif text:  # NEW: Check if text exists, then if query is empty
            st.write("Please enter a question to ask.") # Specific message

    else:
      st.write("RAG service initialization failed.")

else:
    st.write("Please upload a PDF file to get started.")
