 RAG-Powered Q&A Chatbot – Pakistan AI Policy 2025 

This project is a Retrieval-Augmented Generation (RAG) based chatbot built with LangChain. It is designed to answer questions strictly from the contents of an uploaded document. The current implementation uses the "Pakistan National AI Policy 2025" as the source document.

The system retrieves relevant document chunks using FAISS and generates responses with OpenAI’s GPT models. It also includes conversational memory to maintain context across multiple queries.

---

 Features
- PDF document ingestion and parsing  
- Context-aware retrieval using FAISS vector database  
- Conversational memory (remembers the last 5 interactions)  
- Response generation using GPT-3.5-turbo  
- Custom prompt design to ensure concise, document-grounded answers  
- Displays document sources for transparency  

---

 Tech Stack
- Python  
- LangChain  
- FAISS  
- OpenAI API (GPT + embeddings)  
- dotenv  

---

 Project Structure
├── docs/ # Folder containing PDF documents (e.g., Pakistan AI Policy 2025)
├── main.py # Main script for the chatbot
├── .env # Stores the OpenAI API key
├── requirements.txt # Python dependencies
└── README.md # Project documentation
