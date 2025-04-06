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
import os
import shutil
import json
from datetime import datetime

# Create directories for storing files and config
UPLOAD_DIR = Path("uploaded_docs")
CONFIG_DIR = Path("config")
UPLOAD_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.json"

def save_api_key(api_key):
    """Save API key to config file."""
    config = {"openai_api_key": api_key}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def load_api_key():
    """Load API key from config file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("openai_api_key", "")
    except Exception:
        return ""
    return ""

def remove_api_key():
    """Remove the API key from config file."""
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return True
    except Exception:
        return False

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Document GPT", page_icon="📚")
st.title("📚 Document GPT")
st.markdown(
    """
    Welcome! Use this AI chatbot to ask questions about your documents.

    **Instructions:**
    1. Input your OpenAI API Key (required).
    2. Upload one or more **PDFs, TXT, or DOCX** files.
    3. Ask questions related to the uploaded files.
    """
)

# ---- Function to Read Different File Types ----
def read_file(file_path):
    """Read content from different file types."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.suffix.lower() == '.pdf':
        doc = fitz.open(file_path)
        return "\n".join([page.get_text("text") for page in doc])
    elif file_path.suffix.lower() in ['.docx', '.doc']:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    new_filename = f"{timestamp}_{uploaded_file.name}"
    file_path = UPLOAD_DIR / new_filename
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def load_existing_files():
    """Load existing files from the upload directory."""
    if not UPLOAD_DIR.exists():
        return []
    
    files = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.lower()[1:]  # Remove the dot
            })
    return files

def remove_file(file_path):
    """Remove a file from storage."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
        return True
    except Exception as e:
        st.error(f"Error removing file: {e}")
        return False

@st.cache_resource(show_spinner="📖 Processing documents...")
def embed_files(files, openai_api_key):
    """Embed multiple documents and return a retriever."""
    docs = []
    for file in files:
        content = read_file(file["path"])
        if content:
            splitter = CharacterTextSplitter(separator="\n", chunk_size=600, chunk_overlap=100)
            docs.extend(splitter.create_documents([content]))
    
    if not docs:
        return None
    
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

# Initialize session state for stored files and API key
if "stored_files" not in st.session_state:
    st.session_state.stored_files = load_existing_files()
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = load_api_key()

# ---- Sidebar for API Key & File Uploads ----
with st.sidebar:
    api_key_input = st.text_input(
        "🔑 OpenAI API Key", 
        value=st.session_state.openai_api_key,
        type="password",
        key="api_key_input"
    ).strip()
    
    # Update the API key in session state and save to file when changed
    if api_key_input != st.session_state.openai_api_key:
        if api_key_input:  # Only save if api key is not empty
            # Basic validation of API key format
            if len(api_key_input) < 20:  # OpenAI API keys are typically longer
                st.error("Please enter a valid OpenAI API key.")
            else:
                st.session_state.openai_api_key = api_key_input
                save_api_key(api_key_input)
                st.success("API Key saved successfully!")
        elif st.session_state.openai_api_key:  # If there was a key before and now it's empty
            if remove_api_key():
                st.session_state.openai_api_key = ""
                st.success("API Key removed successfully!")
                st.rerun()
    
    st.markdown("### 📂 Upload New Files")
    new_files = st.file_uploader(
        "Upload your files (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if new_files:
        for file in new_files:
            file_path = save_uploaded_file(file)
            # Check if file already exists in stored_files
            if not any(stored_file["path"] == file_path for stored_file in st.session_state.stored_files):
                st.session_state.stored_files.append({
                    "name": file.name,
                    "path": file_path,
                    "type": Path(file.name).suffix.lower()[1:]
                })
        st.success(f"Successfully uploaded {len(new_files)} file(s)!")
    
    st.markdown("### 📚 Loaded Documents")
    files_to_remove = []  # Track files to remove
    
    if st.session_state.stored_files:
        for idx, file in enumerate(st.session_state.stored_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {file['name']}")
            with col2:
                if st.button("🗑️", key=f"remove_{idx}"):
                    if remove_file(file["path"]):
                        files_to_remove.append(idx)
        
        # Remove files after the loop
        if files_to_remove:
            st.session_state.stored_files = [
                f for i, f in enumerate(st.session_state.stored_files) 
                if i not in files_to_remove
            ]
            st.rerun()
    else:
        st.info("No documents loaded yet.")
    
    if st.session_state.stored_files:
        if st.button("🗑️ Clear All Documents", key="clear_all"):
            success = True
            for file in st.session_state.stored_files:
                if not remove_file(file["path"]):
                    success = False
            
            if success:
                st.session_state.stored_files = []
                # Clear the chat history when all documents are removed
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()

if st.session_state.openai_api_key and st.session_state.stored_files:
    openai.api_key = st.session_state.openai_api_key
    retriever = embed_files(st.session_state.stored_files, st.session_state.openai_api_key)
    
    if retriever:
        llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=st.session_state.openai_api_key)

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
        st.warning("No content could be extracted from the uploaded documents. Please try different files.")
else:
    if not st.session_state.openai_api_key:
        st.warning("Please enter an OpenAI API Key.")
    if not st.session_state.stored_files:
        st.warning("Please upload at least one document.")
