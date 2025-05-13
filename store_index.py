from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot"


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",  
            region="us-east-1" 
        )
    )

docssearch = PineconeVectorStore.from_documents( 
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Indexation terminée avec succès.")
