# LangChain Utilities Documentation
This repository contains Python scripts showcasing the integration of various LangChain functionalities for PDF processing, building a chatbot, and implementing Maximum Marginal Relevance (MMR) search. The LangChain library is utilized for natural language processing tasks, including embeddings, vector storage, and chat models.

## Table of Contents
- PDF Processing
- Chatbot Setup
- Maximum Marginal Relevance (MMR) Search
- Dependencies
- Running the Scripts
- Contributing
- License
### PDF Processing
#### Code Overview
- The PDF processing script (documentLoadingSpliting/load_n_split.py) demonstrates the LangChain library's usage to load and split a PDF document into pages.
- Additionally, it leverages SentenceTransformer embeddings and Chroma vector storage to create a persistent vector store.

#### Dependencies
- LangChain: A library for natural language processing tasks.
- SentenceTransformers: A library for sentence embeddings.
- PyPDF2: A library for reading PDF files.
#### Usage
- Set the PDF file path in the FILE_PATH variable inside the script.
- Run the script using python process_pdf.py.
### Rich Dad Poor Dad Chatbot Setup
#### Code Overview
The chatbot setup script (chatbot/app.py) showcases the creation of a conversational retrieval chain using LangChain and Streamlit. It combines SentenceTransformer embeddings, Chroma vector storage, and the OpenAI model to build a chatbot capable of answering user questions about the book "Rich Dad Poor Dad."

#### Dependencies
- LangChain: A library for natural language processing tasks.
- LangChain Community: Extended functionality for LangChain.
- SentenceTransformers: A library for sentence embeddings.
- Python-Decouple: A library for handling configuration variables.
```bash
pip install langchain sentence-transformers PyPDF2 python-decouple chromadb streamlit
```
#### Usage
- Set up your OpenAI API key in the environment or in a .env file.
- Run the chatbot using ``` streamlit run app.py```

### Maximum Marginal Relevance (MMR) Search
#### Code Overview
The MMR search script (retrievalTechniques/mmr.py) exemplifies LangChain's MMR capabilities. It utilizes SentenceTransformer embeddings, Chroma vector storage, and MMR search functionality to provide relevant responses to a user query.


#### Dependencies
Make sure to install the required dependencies for each script. You can install them using:

```bash
pip install langchain sentence-transformers PyPDF2 python-decouple chromadb
```
#### Usage
- Set up your OpenAI API key in the environment or in a .env file.
- Run the script using ```python mmr.py```


### Contributing
Contributions to this project are welcome. Follow the guidelines in CONTRIBUTING.md for details on how to contribute.

### License
This project is licensed under the MIT License.





