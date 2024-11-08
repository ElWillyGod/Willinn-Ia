from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

# Cargar los embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cargar documentos y vectorizarlos
loader = TextLoader('/home/kali/Downloads/char1.txt')
documents = loader.load()
doc_texts = [doc.page_content for doc in documents]

# Crear el índice FAISS
vector_store = FAISS.from_texts(doc_texts, embeddings)
