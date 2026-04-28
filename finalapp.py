import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API Key
os.environ["NVIDIA_API_KEY"]=os.getenv("NVIDIA_API_KEY")

llm=ChatNVIDIA(model="openai/gpt-oss-20b")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Nvidia Nim demo")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the queqtion based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

prompt1=st.text_input("Enter your question from documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB is ready")

if prompt1:
    #document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    #retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    #response=retrieval_chain.invoke({"input":prompt1})
    # Récupérer les documents
    docs = retriever.invoke(prompt1)
    # Formatter le contexte 
    context = "\n\n".join(doc.page_content for doc in docs)
    # Construire la chaîne RAG
    rag_chain = (
        {"context": lambda x: context, "input": RunnablePassthrough()}
        | prompt
        | llm
    )
    response = rag_chain.invoke(prompt1)

    print("Response time:",time.process_time()-start)
    #st.write(response["answer"])
    st.write(response.content)
    # With streamlit expander
    with st.expander("Document similarity search"):
        # Find the relevant chunks
        for i,doc in enumerate(docs):
            st.write(doc.page_content)
            st.write("---------------------")