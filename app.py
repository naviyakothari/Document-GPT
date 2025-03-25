# First version of the app

import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
import fitz  # PyMuPDF
import docx


# ---- Streamlit Page Config ----
st.set_page_config(page_title="Document GPT", page_icon="📚")
st.title("📚 Document GPT")
st.markdown(
    """
    Welcome! Upload one or more documents and ask AI questions about them.

    **Instructions:**
    1. Input your OpenAI API Key (required).
    2. Upload one or more **PDFs, TXT, or DOCX** files.
    3. Ask questions related to the uploaded files.
    """
)

# ---- Function to Read Different File Types ----
def read_file(file):
    """Read content from different file types."""
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

@st.cache_resource(show_spinner="📖 Processing documents...")
def embed_files(files, openai_api_key):
    """Embed multiple documents and return a retriever."""
    docs = []
    for file in files:
        content = read_file(file)
        if content:
            splitter = CharacterTextSplitter(separator="\n", chunk_size=600, chunk_overlap=100)
            docs.extend(splitter.create_documents([content]))
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

# ---- Format Documents for Context ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Answer using ONLY the following context. If you don't know the answer, say so.
        Context: {context}"""),
        ("human", "{question}"),
    ]
)

# ---- Sidebar for API Key & File Uploads ----
with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password").strip()
    files = st.file_uploader(
        "📂 Upload your files (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

if openai_api_key and files:
    openai.api_key = openai_api_key
    retriever = embed_files(files, openai_api_key)
    llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=openai_api_key)

    st.session_state.setdefault("messages", [])
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["message"])
    
    message = st.chat_input("Ask anything about the uploaded documents...")
    if message:
        with st.chat_message("human"):
            st.markdown(message)
        st.session_state["messages"].append({"message": message, "role": "human"})

        chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm)
        with st.chat_message("ai"):
            response = chain.invoke(message).content
            st.markdown(response)
        st.session_state["messages"].append({"message": response, "role": "ai"})
else:
    st.warning("Please enter an OpenAI API Key and upload at least one document.")
