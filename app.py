# app.py
import streamlit as st
from rag import RAGService
from summarization import generate_summary
from preprocessing import extract_text_from_pdf
import io
import hashlib

# Function to generate a unique key based on file content
def generate_file_key(uploaded_file):
    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_contents).hexdigest()
        return file_hash
    return None

# Function to handle file upload and processing, NOW CACHED
@st.cache_data() # VERY IMPORTANT: Cache this function!
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with st.spinner("Processing the PDF..."):
            bytes_data = uploaded_file.getvalue()
            text = extract_text_from_pdf(io.BytesIO(bytes_data))
        return text
    return None

# --- App UI ---
st.title("PDF Question Answering with RAG")
st.write("Upload a PDF and ask questions about it.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
file_key = generate_file_key(uploaded_file)  # Get the unique file key
text = process_uploaded_file(uploaded_file) # Get extracted text


# Use st.cache_resource with a key based on the file.
@st.cache_resource(show_spinner=False)
def get_rag_service(key):  # Only the KEY is needed now
    if key is not None: # Only create if key exists.
        text = process_uploaded_file(uploaded_file) #Get text again
        return RAGService(text)
    return None

if file_key: #Use file key instead of text
    rag_service = get_rag_service(key=file_key) # Only the KEY

    if rag_service: # Check if rag service exists
        with st.expander("Show Summary"):
            st.write(generate_summary(rag_service.text))

        user_query = st.text_input("Enter your question:")

        if user_query:
            with st.spinner("Thinking..."):
                answer = rag_service.generate_answer(user_query)
            st.write(answer)
else:
    st.write("Please upload a PDF file to get started.")