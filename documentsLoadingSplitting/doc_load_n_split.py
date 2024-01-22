from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


FILE_PATH = '../documents/Rich-Dad-Poor-Dad.pdf'

# create loader
loader = PyPDFLoader(FILE_PATH)
# split document
pages = loader.load_and_split()

#print(len(pages))

# embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store db
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_function,
    persist_directory=f"../vector_db",
    collection_name="rich_dad_poor_dad")

# make vector store persistant
vectordb.persist()