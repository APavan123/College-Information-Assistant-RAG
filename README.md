# 🎓 College Information Assistant Using RAG

## Project Overview

The College Information Assistant is an AI-powered chatbot that helps students get information about college admissions, courses, fee structures, faculty and facilities.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from college PDF documents and generate accurate answers using Google Gemini.

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS Vector Database
- Google Generative AI Embeddings
- Retrieval-Augmented Generation (RAG)

## RAG Workflow

1. Upload college PDF documents.
2. Extract text from PDF files.
3. Split text into smaller chunks.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant information based on user questions.
7. Generate answers using Gemini.
8. Display answers through Streamlit.

## Features

- College information chatbot.
- PDF-based question answering.
- Semantic search using FAISS.
- Gemini-powered responses.
- Source document references.
- Interactive Streamlit interface.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/College-Information-Assistant-RAG.git

cd College-Information-Assistant-RAG

pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

## Run Application

```bash
streamlit run app.py
```

## Author

Pavan

Aspiring AI/ML Engineer | Generative AI | LangChain | RAG
