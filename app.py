from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from llama_cpp import Llama

app = Flask(__name__)

# Inicializar modelo Llama2 en formato GGML
model_path = "/ruta/al/modelo/llama-2-7b-chat.ggmlv3.q4_0.bin"
llm = Llama(model_path=model_path)

# Cargar embeddings y crear índice FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = TextLoader('ruta_al_documento.txt')
documents = loader.load()
doc_texts = [doc.page_content for doc in documents]
vector_store = FAISS.from_texts(doc_texts, embeddings)

# Configurar el chain de recuperación conversacional
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    history = request.json.get('history', [])
    response = qa_chain.run({"question": user_input, "chat_history": history})
    return jsonify({"response": response})

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    results = vector_store.similarity_search(query, k=5)
    return jsonify({"results": [res.page_content for res in results]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
