"""
RAG Pipeline combining vector store retrieval with Qwen LLM generation.
Includes conversation memory and knowledge learning from interactions.
"""

import re
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_vector_store, create_vector_store
from llm import QwenLLM


def clean_response(text: str) -> str:
    """
    Clean up response while preserving bold formatting (**text**).
    Ensures proper line breaks between bullet points.

    Args:
        text: The raw response text from LLM.

    Returns:
        Cleaned text with bold formatting and proper line breaks.
    """
    if not text:
        return text

    # Remove markdown headers: ### Header → Header
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove inline code: `code` → code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove code blocks: ```code``` → code
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', ''), text)

    # Remove link formatting: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove image formatting: ![alt](url) → alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # FIX: Force line breaks before bullet points
    # This handles the case where model outputs everything on one line
    # Pattern: " - " in the middle of text (not at start) should become "\n- "
    text = re.sub(r'\s+-\s+\*\*', '\n- **', text)  # " - **text**" → "\n- **text**"
    text = re.sub(r'\s+-\s+([A-Z])', r'\n- \1', text)  # " - The" → "\n- The"

    # Handle cases like "standards. - **SGN" → "standards.\n- **SGN"
    text = re.sub(r'([.!?])\s*-\s+', r'\1\n- ', text)

    # Ensure proper newlines before all remaining bullet points
    text = re.sub(r'(?<!\n)-\s+', '\n- ', text)

    # Clean up multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Clean up multiple newlines (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix any double newlines before bullets
    text = re.sub(r'\n\n+- ', '\n- ', text)

    # Ensure first character isn't a newline
    text = text.strip()

    # If text starts with "- ", it's already a bullet point
    if not text.startswith('-'):
        # Check if there are bullet points and add newline after first sentence
        if '\n- ' in text:
            # Find first period followed by bullet and ensure newline
            text = re.sub(r'^([^.]+\.)\s*\n?-', r'\1\n\n-', text)

    return text


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Retrieves relevant documents and generates answers using Qwen 2.5 7B.
    """

    def __init__(self, model_name: str = "qwen2.5:7b", enable_learning: bool = True):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Ollama model name (e.g., qwen2.5:7b).
            enable_learning: Whether to store Q&A pairs for future retrieval.
        """
        print("Initializing RAG Pipeline...")
        self.vectorstore = get_vector_store()
        self.llm = QwenLLM(model_path=model_name)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        # Conversation memory - stores Q&A history for current session
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        self.enable_learning = enable_learning
        print("RAG Pipeline initialized!")

    # Keywords that indicate a follow-up/contextual question
    FOLLOWUP_INDICATORS = [
        "above", "this", "that", "it", "the section", "the problem",
        "the issue", "the error", "the topic", "the same", "previous",
        "you mentioned", "you said", "explain more", "tell me more",
        "elaborate", "clarify", "what do you mean", "can you explain",
        "another way", "different way", "simpler", "in detail", "more detail",
        "how to fix", "how to solve", "how to resolve", "what about",
        "and also", "furthermore", "additionally", "related to"
    ]

    def _is_followup_question(self, question: str) -> bool:
        """
        Detect if the question is a follow-up that references previous context.

        Args:
            question: The user's question.

        Returns:
            True if it's a follow-up question.
        """
        question_lower = question.lower()

        # Check for follow-up indicators
        for indicator in self.FOLLOWUP_INDICATORS:
            if indicator in question_lower:
                return True

        # Short questions are often follow-ups (e.g., "explain more", "why?")
        if len(question.split()) <= 5 and self.conversation_history:
            return True

        return False

    def _get_enhanced_query(self, question: str) -> str:
        """
        Enhance a vague follow-up question with context from conversation history.

        Args:
            question: The original question.

        Returns:
            Enhanced question with context.
        """
        if not self.conversation_history:
            return question

        # Get the last exchange for context
        last_exchange = self.conversation_history[-1]
        last_question = last_exchange["question"]

        # Create an enhanced query that includes context
        enhanced_query = f"{last_question} {question}"

        return enhanced_query

    def query(
        self,
        question: str,
        num_docs: int = 6,
        max_new_tokens: int = 2048,
        use_history: bool = True
    ) -> dict:
        """
        Query the RAG system with conversation memory and learning.
        Handles follow-up questions like "explain the above" or "how to fix this".

        Args:
            question: The user's question.
            num_docs: Number of documents to retrieve.
            max_new_tokens: Maximum tokens in the response.
            use_history: Whether to include conversation history.

        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Check if this is a follow-up question
        is_followup = self._is_followup_question(question)

        # For follow-up questions, use enhanced query for better retrieval
        search_query = question
        if is_followup and self.conversation_history:
            search_query = self._get_enhanced_query(question)

        # Retrieve relevant documents using the (potentially enhanced) query
        docs = self.retriever.invoke(search_query)

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

        # Build conversation history string - include more context for follow-ups
        history_str = ""
        last_answer_context = ""

        if use_history and self.conversation_history:
            history_parts = []
            # For follow-up questions, include more history
            history_limit = self.max_history if is_followup else min(5, self.max_history)

            for exchange in self.conversation_history[-history_limit:]:
                history_parts.append(f"User: {exchange['question']}\nAssistant: {exchange['answer']}")
            history_str = "\n\n".join(history_parts)

            # Get the last answer specifically for follow-up context
            if is_followup:
                last_answer_context = self.conversation_history[-1]["answer"]

        # Build the prompt with enhanced context awareness
        prompt = f"""You are a helpful assistant. Format ALL responses using bullet points with bold key terms.

FORMATTING RULES - MUST FOLLOW:

1. Use bullet points (-) for ALL information - no paragraphs
2. Use **bold** for key terms, technical words, names, and important concepts
3. Break long sentences into multiple bullet points
4. Use nested bullet points for sub-items (indent with spaces)
5. Keep each bullet point concise and scannable

EXAMPLE FORMAT:

- The **Self-Lay Operator (SLO)** plays a crucial role in gas network projects
- Main responsibilities include:
  - **Load Enquiry**: Supplying all required details to SGN
  - **Feasibility Studies**: Evaluating technical and economic viability
  - **Design**: Developing conceptual and detailed designs
- The SLO must comply with **SGN standards** and **industry regulations**
- Key phases of the audit process:
  - **Phase 1**: Initial assessment
  - **Phase 2**: Detailed review
  - **Phase 3**: Final approval

WHAT TO MAKE BOLD:
- Technical terms (e.g., **Conceptual Design**, **Feasibility Study**)
- Names and abbreviations (e.g., **SGN**, **SLO**, **Phase 3**)
- Important concepts (e.g., **safety requirements**, **compliance**)
- Key actions (e.g., **must submit**, **required to**)

CONTEXT RULES:
- If user says "above", "this", "that", "it" - refer to previous conversation
- If user says "explain more" - elaborate on previous topic
- If answer not found - say "I could not find the answer in the provided documents."

Knowledge Base Context:
{context}
"""

        if history_str:
            prompt += f"""
=== CONVERSATION HISTORY (Use this to understand references like "above", "this", "it") ===
{history_str}
=== END OF CONVERSATION HISTORY ===

"""

        if is_followup and last_answer_context:
            prompt += f"""
Note: The user's question appears to be a follow-up. The most recent topic discussed was:
"{last_answer_context[:500]}..."

"""

        prompt += f"""Current Question: {question}

Answer (use bullet points with **bold** for key terms):"""

        # Generate response
        response = self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.5
        )

        # Post-process to clean up while preserving bold formatting
        response = clean_response(response)

        # Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": response,
            "timestamp": datetime.now().isoformat(),
            "is_followup": is_followup
        })

        # Learn from interaction - store Q&A in vector store for future retrieval
        # For follow-ups, store with the context of what was being discussed
        if self.enable_learning and response and "could not find" not in response.lower():
            if is_followup and self.conversation_history and len(self.conversation_history) > 1:
                # Store with context from previous question
                prev_question = self.conversation_history[-2]["question"]
                contextualized_question = f"{prev_question} - {question}"
                self._store_learned_knowledge(contextualized_question, response)
            else:
                self._store_learned_knowledge(question, response)

        return {
            "answer": response,
            "sources": sources,
            "num_docs_retrieved": len(docs),
            "history_used": len(self.conversation_history) - 1,
            "detected_followup": is_followup
        }

    def _store_learned_knowledge(self, question: str, answer: str) -> None:
        """
        Store Q&A pair in vector store for future retrieval.

        Args:
            question: The user's question.
            answer: The generated answer.
        """
        # Create a document combining Q&A for future retrieval
        learned_content = f"""Topic: {question}

Explanation: {answer}"""

        doc = Document(
            page_content=learned_content,
            metadata={
                "source": "learned_interaction",
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "type": "qa_pair"
            }
        )

        # Add directly to vector store (no splitting needed for Q&A pairs)
        self.vectorstore.add_documents([doc])

    def clear_history(self) -> dict:
        """
        Clear the conversation history.

        Returns:
            Status dictionary.
        """
        self.conversation_history = []
        return {
            "status": "success",
            "message": "Conversation history cleared"
        }

    def get_history(self) -> list:
        """
        Get the current conversation history.

        Returns:
            List of conversation exchanges.
        """
        return self.conversation_history

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
            search_kwargs={"k": 6}
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
