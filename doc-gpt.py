import os
import openai
import pypdf
import streamlit as st
from langchain.llms import OpenAIChat
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

@st.cache_data
def split_pdf(fpath,chunk_chars=4000,overlap=50):
    """
    Pre-process PDF into chunks
    Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    """
    st.info("`Reading and splitting doc ...`")
    pdfReader = pypdf.PdfReader(fpath)
    splits = []
    split = ""
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        while len(split) > chunk_chars:
            splits.append(split[:chunk_chars])
            split = split[chunk_chars - overlap :]
    if len(split) > overlap:
        splits.append(split[:chunk_chars])
    return splits

@st.cache_resource
def create_ix(splits):
    """ 
    Create vector DB index of PDF
    """
    st.info("`Building index ...`")
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(splits,embeddings)

# Auth
st.sidebar.image("Img/reading.jpg")
api_key = st.sidebar.text_input("`OpenAI API Key:`", type="password")
st.sidebar.write("`By:` [@RLanceMartin](https://twitter.com/RLanceMartin)")
os.environ["OPENAI_API_KEY"] = api_key
chunk_chars = st.sidebar.radio("`Choose chunk size for splitting`", (2000, 3000, 4000), index=1)
st.sidebar.info("`Larger chunk size can produce better answers, but may hit ChatGPT context limit (4096 tokens)`")

# App 
st.header("`doc-gpt`")
st.info("`Hello! I am a ChatGPT connected to whatever document you upload.`")
uploaded_file_pdf = st.file_uploader("`Upload PDF File:` ", type = ['pdf'] , accept_multiple_files=False)
if uploaded_file_pdf and api_key:
    # Split and create index
    d=split_pdf(uploaded_file_pdf,chunk_chars)
    if d:
        ix=create_ix(d)
        # Use ChatGPT with index QA chain
        llm = OpenAIChat(temperature=0)
        chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=ix)
        query = st.text_input("`Please ask a question:` ","What is this document about?")
        try:
            st.info(f"`{chain.run(query)}`")
        except openai.error.InvalidRequestError:
            # Limitation w/ ChatGPT: 4096 token context length
            # https://github.com/acheong08/ChatGPT/discussions/649
            st.warning('Error with model request, often due to context length. Try reducing chunk size.', icon="⚠️")
    else:
        st.warning('Error with reading pdf, often b/c it is a scanned image of text. Try another file.', icon="⚠️")

else:
    st.info("`Please enter OpenAI Key and upload pdf file`")
