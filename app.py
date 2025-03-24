# First version of the app

# import streamlit as st
# import openai
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# from langchain.vectorstores.faiss import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from pathlib import Path
# import fitz  # PyMuPDF
# import docx

# # Set up Streamlit app
# st.set_page_config(page_title="Document GPT", page_icon="📚")
# st.title("Document GPT")

# st.markdown(
#     """
# Welcome! Upload multiple documents and chat with AI.

# 1. Enter your OpenAI API Key.
# 2. Upload one or more documents (.txt, .pdf, .docx).
# 3. Ask questions about the uploaded files.
#     """
# )

# def read_file(file):
#     """Read content from different file types."""
#     if file.type == "text/plain":
#         return file.read().decode("utf-8")
#     elif file.type == "application/pdf":
#         doc = fitz.open(stream=file.read(), filetype="pdf")
#         return "\n".join([page.get_text("text") for page in doc])
#     elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#         doc = docx.Document(file)
#         return "\n".join([para.text for para in doc.paragraphs])
#     return ""

# def embed_files(files, openai_api_key):
#     """Embed multiple documents and return a retriever."""
#     docs = []
#     for file in files:
#         content = read_file(file)
#         if content:
#             splitter = CharacterTextSplitter(separator="\n", chunk_size=600, chunk_overlap=100)
#             docs.extend(splitter.create_documents([content]))
    
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     return vectorstore.as_retriever()

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """Answer using ONLY the following context. If you don't know the answer, say so.
#         Context: {context}"""),
#         ("human", "{question}"),
#     ]
# )

# # Sidebar for API key and file uploads
# with st.sidebar:
#     openai_api_key = st.text_input("Enter your OpenAI API Key", type="password").strip()
#     # st.write(f"API Key: {openai_api_key[:5]}...{openai_api_key[-5:]}")  # Masked for security
#     files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# if openai_api_key and files:
#     openai.api_key = openai_api_key
#     retriever = embed_files(files, openai_api_key)
#     llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=openai_api_key)

#     st.session_state.setdefault("messages", [])
#     for msg in st.session_state["messages"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["message"])
    
#     message = st.chat_input("Ask anything about the uploaded documents...")
#     if message:
#         with st.chat_message("human"):
#             st.markdown(message)
#         st.session_state["messages"].append({"message": message, "role": "human"})

#         chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm)
#         with st.chat_message("ai"):
#             response = chain.invoke(message)
#             st.markdown(response)
#         st.session_state["messages"].append({"message": response, "role": "ai"})
# else:
#     st.warning("Please enter your OpenAI API key and upload files.")




# 2nd version of the app

import streamlit as st
import openai
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import fitz  # PyMuPDF for PDFs
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

# ---- Sidebar for API Key & File Uploads ----
with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
    uploaded_files = st.file_uploader(
        "📂 Upload your files (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

# ---- Store Chat History ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---- Function to Read Different File Types ----
def read_file(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:  # TXT files
        text = file.read().decode("utf-8")
    return text

# ---- Function to Embed Files & Create a Retriever ----
@st.cache_resource(show_spinner="📖 Processing documents...")
def embed_files(files, api_key):
    docs = []
    for file in files:
        content = read_file(file)
        splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100, separator="\n")
        docs.extend(splitter.create_documents([content]))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore.as_retriever()  # ✅ Now cached properly!

# ---- Format Documents for Context ----
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ---- Streamlit Chat UI ----
def display_message(message, role):
    with st.chat_message(role):
        st.markdown(message)

# ---- Show Chat History ----
for message in st.session_state["messages"]:
    display_message(message["message"], message["role"])

# ---- Chat Handling ----
if openai_api_key and uploaded_files:
    retriever = embed_files(uploaded_files, openai_api_key)

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=openai_api_key,
        callbacks=[StreamingStdOutCallbackHandler()],  # Stream response smoothly
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use the following context to answer questions:\n\n{context}"),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    # ---- Chat Input ----
    user_query = st.chat_input("Ask a question about the uploaded documents...")
    if user_query:
        display_message(user_query, "human")
        st.session_state["messages"].append({"message": user_query, "role": "human"})

        with st.chat_message("ai"):
            response = chain.invoke(user_query)
            st.markdown(response.content)
            st.session_state["messages"].append({"message": response.content, "role": "ai"})

else:
    st.warning("Please enter an OpenAI API Key and upload at least one document.")