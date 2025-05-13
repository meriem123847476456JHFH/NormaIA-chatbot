import streamlit as st
from src.helper import download_hugging_face_embeddings
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import *
import os

# Chargement des variables d'environnement
load_dotenv()

# Initialisation des clés API
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Téléchargement des embeddings
embeddings = download_hugging_face_embeddings()

# Connexion à l'index Pinecone
index_name = "chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Création du retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 30})

# Initialisation du LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
)

# Création des prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Création de la chaîne RAG
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Interface Streamlit

st.set_page_config(page_title="NormaIA", page_icon="🤖")
st.title("💬Smart NormaIA– Guide des Normes & Réglementations")

# Session state pour garder l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Champ d'entrée utilisateur
if prompt := st.chat_input("Posez votre question sur les normes d'export..."):
    # Afficher la question utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Générer la réponse via RAG
    response = rag_chain.invoke({"input": prompt})
    answer = response["answer"]

    # Afficher la réponse IA
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
