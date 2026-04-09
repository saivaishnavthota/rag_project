"""
RAG Pipeline combining vector store retrieval with Qwen LLM generation.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_vector_store, create_vector_store
from llm import QwenLLM


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Retrieves relevant documents and generates answers using Qwen 2.5 7B.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the RAG pipeline.

        Args:
            model_path: Path to the Qwen model (HuggingFace ID or local path).
        """
        print("Initializing RAG Pipeline...")
        self.vectorstore = get_vector_store()
        self.llm = QwenLLM(model_path=model_path)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("RAG Pipeline initialized!")

    def query(
        self,
        question: str,
        num_docs: int = 4,
        max_new_tokens: int = 512
    ) -> dict:
        """
        Query the RAG system.

        Args:
            question: The user's question.
            num_docs: Number of documents to retrieve.
            max_new_tokens: Maximum tokens in the response.

        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Build context from retrieved documents
        context_parts = []
        sources = []

        for i, doc in enumerate(docs):
            context_parts.append(f"[Document {i+1}]\n{doc.page_content}")
            sources.append({
                "content_preview": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })

        context = "\n\n".join(context_parts)

        # Build the prompt
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not found in the context, say "I could not find the answer in the provided documents."
Be concise and accurate in your response.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        response = self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )

        return {
            "answer": response,
            "sources": sources,
            "num_docs_retrieved": len(docs)
        }

    def add_document(self, content: str, metadata: dict = None) -> dict:
        """
        Add a new document to the vector store.

        Args:
            content: The document text content.
            metadata: Optional metadata dictionary.

        Returns:
            Status dictionary.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        doc = Document(
            page_content=content,
            metadata=metadata or {}
        )

        splits = text_splitter.split_documents([doc])
        self.vectorstore.add_documents(splits)

        return {
            "status": "success",
            "chunks_added": len(splits)
        }

    def add_documents_batch(self, documents: list) -> dict:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'.

        Returns:
            Status dictionary.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        all_docs = []
        for doc_data in documents:
            doc = Document(
                page_content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            all_docs.append(doc)

        splits = text_splitter.split_documents(all_docs)
        self.vectorstore.add_documents(splits)

        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_added": len(splits)
        }

    def reindex(self) -> dict:
        """
        Reindex all documents from the documents directory.

        Returns:
            Status dictionary.
        """
        print("Reindexing documents...")
        self.vectorstore = create_vector_store()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return {
            "status": "success",
            "message": "Documents reindexed successfully"
        }

    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Statistics dictionary.
        """
        collection = self.vectorstore._collection
        count = collection.count()

        return {
            "total_chunks": count,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "Qwen/Qwen2.5-7B-Instruct"
        }
