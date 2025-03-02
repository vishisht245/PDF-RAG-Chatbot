import google.generativeai as genai
import os
import dotenv
from preprocessing import extract_text_from_pdf

def generate_summary(text):
    """
    Generates a summary of the given text using Gemini.

    Args:
        text: The text to summarize.

    Returns:
        The generated summary.
    """
    dotenv.load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""Summarize the following text in a concise and informative way,
              capturing the main points:

              {text}"""
    response = model.generate_content(prompt)
    return response.text

if __name__ == '__main__':
    pdf_file = "The_Gift_of_the_Magi.pdf"
    text = extract_text_from_pdf(pdf_file)
    summary = generate_summary(text)
    print(summary)
