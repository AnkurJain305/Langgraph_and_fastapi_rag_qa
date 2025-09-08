from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)
