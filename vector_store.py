"""
Vector store management using ChromaDB.
Handles document loading, chunking, and persistence.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from embeddings import get_embeddings

# Configuration
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def load_documents():
    """
    Load all documents from the documents directory.
    Supports PDF, TXT, and DOCX files.
    """
    loaders = [
        DirectoryLoader(
            DOCS_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        DirectoryLoader(
            DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            loader_kwargs={"autodetect_encoding": True},
        ),
        DirectoryLoader(
            DOCS_DIR,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
        ),
    ]

    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {loader.__class__.__name__}")
        except Exception as e:
            print(f"Error loading documents: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


def create_vector_store():
    """
    Create a new vector store from documents in the documents directory.
    """
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found. Creating empty vector store.")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings()
        )
        return vectorstore

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR
    )
    print("Vector store created and persisted.")

    return vectorstore


def get_vector_store():
    """
    Get existing vector store or create a new one if it doesn't exist.
    """
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings()
        )

    print("No existing vector store found. Creating new one...")
    return create_vector_store()


def add_documents_to_store(documents: list):
    """
    Add new documents to the existing vector store.
    """
    vectorstore = get_vector_store()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)

    vectorstore.add_documents(splits)
    print(f"Added {len(splits)} chunks to vector store")

    return vectorstore
