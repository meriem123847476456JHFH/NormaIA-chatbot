
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

#Extract Data From the PDF FILE
def load_pdf_file(data):
    documents = []
    for filename in os.listdir(data):
        if filename.endswith(".pdf"):
            path = os.path.join(data, filename)
            print(f"🔍 Traitement du fichier : {filename}")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                documents.extend(docs)
                
            except Exception as e:
                print(f"❌ Erreur avec {filename} : {e}")
                with open("load_errors.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{filename} : {e}\n")
    return documents                                         

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={"device": "cpu"}  # 👈 Très important !
    )
    return embeddings

