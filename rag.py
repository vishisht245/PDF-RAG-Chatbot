# rag.py
import google.generativeai as genai
import os
import dotenv
# from preprocessing import extract_text_from_pdf  # No longer needed here
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

class RAGService:
    def __init__(self, text):
        dotenv.load_dotenv()
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.text = text
        self.chunks = self.chunk_text(self.text)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.create_collection()
        self.add_to_collection(self.chunks)


    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def create_collection(self):
        # Use get_or_create_collection:
        collection = self.client.get_or_create_collection("my_collection")
        return collection

    def add_to_collection(self, chunks):
      # Get existing IDs from the collection
        existing_ids = set(self.collection.get()['ids'])

        embeddings = []
        documents = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(i)
            # Only add the chunk if its ID doesn't already exist
            if chunk_id not in existing_ids:
                embeddings.append(self.embedding_model.encode(chunk).tolist())
                documents.append(chunk)
                ids.append(chunk_id)
        # Add only the new chunks to the collection
        if ids: # Check if there are any new items.
          self.collection.add(embeddings=embeddings, documents=documents, ids=ids)



    def retrieve_relevant_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results['documents'][0]

    def generate_answer(self, query):
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

if __name__ == '__main__':
    #For testing, read a text file
    with open("test.txt", "r") as f: #Create a simple text file for testing.
      test_text = f.read()
    rag_service = RAGService(test_text)
    user_query = "What did Della sell to buy Jim a gift?"
    answer = rag_service.generate_answer(user_query)
    print(answer)

    user_query = "What is the capital of France?"
    answer = rag_service.generate_answer(user_query)
    print(answer)