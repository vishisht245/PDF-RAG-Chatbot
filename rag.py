# rag.py
import google.generativeai as genai
import os
import dotenv
from sentence_transformers import SentenceTransformer  # No 'exceptions'
import chromadb
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGService:
    def __init__(self, text):
        dotenv.load_dotenv()
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

        if not GOOGLE_API_KEY:
            logging.error("Google API key not found.")
            raise ValueError("Missing Google API Key")

        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logging.error(f"Error configuring Gemini API: {e}")
            raise

        if not text:
            logging.error("Input text cannot be empty.")
            raise ValueError("Input text cannot be empty")

        self.text = text
        self.chunks = self.chunk_text(self.text)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:  # Catch the base exception class
            logging.error(f"Error loading Sentence Transformer model: {e}")
            raise

        # Create a NEW Chroma client instance *inside* __init__:
        self.client = chromadb.Client()  # In-memory client
        self.collection = self.create_collection()  # Get or create
        self.add_to_collection(self.chunks)


    def chunk_text(self, text, chunk_size=500, overlap=50):
        try:
            if not isinstance(text, str):
                raise TypeError("Input text must be a string")
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks
        except TypeError as e:
            logging.error(f"Error in chunk_text: {e}")
            raise

    def create_collection(self):
        try:
            # Use get_or_create_collection:
            collection = self.client.get_or_create_collection("my_collection")
            return collection
        except Exception as e:
            logging.error(f"Error creating/getting ChromaDB collection: {e}")
            raise
    def add_to_collection(self, chunks):
        try:
            existing_ids = set(self.collection.get()['ids'])
            embeddings = []
            documents = []
            ids = []

            for i, chunk in enumerate(chunks):
                chunk_id = str(i)
                if chunk_id not in existing_ids:
                    embeddings.append(self.embedding_model.encode(chunk).tolist())
                    documents.append(chunk)
                    ids.append(chunk_id)

            if ids:
                self.collection.add(embeddings=embeddings, documents=documents, ids=ids)
        except Exception as e:
            logging.error(f"Error adding to ChromaDB collection: {e}")
            raise

    def retrieve_relevant_chunks(self, query, top_k=3):
        try:
            if not isinstance(query, str):
                raise TypeError("Query should be string")
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            return results['documents'][0]
        except TypeError as e:
          logging.error(f"Error in retrieve_relevant_chunks: {e}")
          raise
        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {e}")
            raise

    def generate_answer(self, query):
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
            prompt = f"""Answer the following question based on the context provided but don't mention it, keep the tone friendly and warm and answer with confidence:
                        Question: {query}
                        Context:
                        {context}

                        If the answer cannot be found in the context, respond with 'I am sorry, but I don't have enough information to answer that question from the context I was given.'
                        """
            response = self.model.generate_content(prompt)
            return response.text
        except google.api_core.exceptions.GoogleAPIError as e:
            logging.error(f"Gemini API error in generate_answer: {e}")
            return "I encountered an error connecting to the Gemini API."
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise
if __name__ == '__main__':
    try:
      with open("test.txt", "r") as f:
        test_text = f.read()
      rag_service = RAGService(test_text)
      user_query = "What did Della sell to buy Jim a gift?"
      answer = rag_service.generate_answer(user_query)
      print(answer)

      user_query = "What is the capital of France?"
      answer = rag_service.generate_answer(user_query)
      print(answer)
    except Exception as e:
      print("Error: ", e)
