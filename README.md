# PDF Question Answering with RAG (Retrieval-Augmented Generation)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content.  It uses Streamlit for the web interface, Google Gemini for OCR and text generation, and ChromaDB for vector storage.

## Features

*   **PDF Upload:** Users can upload their own PDF files.
*   **OCR:** Extracts text from PDFs using Google Gemini's vision capabilities (handles scanned documents).
*   **Summarization:** Generates a concise summary of the uploaded document.
*   **Question Answering:** Answers questions based on the PDF content using a RAG approach.
*   **Vector Database:** Uses ChromaDB for efficient similarity search and retrieval of relevant text chunks.
*   **Web Interface:** Provides a simple, interactive web interface with Streamlit.

## Dependencies

*   PyMuPDF (fitz)
*   google-generativeai
*   sentence-transformers
*   chromadb
*   streamlit
*   python-dotenv
*   Pillow (PIL)
*   numpy

These can be installed using: `pip install -r requirements.txt`

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Obtain a Google Gemini API Key:**
    *   Go to [Google AI Studio](https://ai.google.dev/).
    *   Create a new project (or use an existing one).
    *   Create an API Key.

3.  **Set the API Key:**
    *   Create a `.env` file in the project's root directory:
        ```
        GOOGLE_API_KEY="your-api-key-here"
        ```
        Replace `"your-api-key-here"` with your actual API key.

## Running the Application

1.  Make sure you are in your virtual environment (activate it if necessary).
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  A new tab will open in your web browser with the application.  Upload a PDF and start asking questions!

## Project Structure
rag-pdf-qa/ (or your project's name)
├── app.py # Streamlit web interface
├── preprocessing.py # PDF preprocessing (OCR) and text extraction
├── rag.py # RAG service (chunking, embedding, retrieval)
├── summarization.py # Text summarization
├── requirements.txt # Project dependencies
├── .env # API Key (DO NOT COMMIT THIS TO GITHUB)
└── README.md # This file
