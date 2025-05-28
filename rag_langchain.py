#!/usr/bin/env python3
"""
LangChain-based RAG implementation for cardiology knowledge evaluation.
Uses gpt-4o-mini for cost efficiency.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import time

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import OPENAI_API_KEY

class LangChainRAGEvaluator:
    """LangChain-based RAG system for cardiology knowledge evaluation."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini", 
                 pdfs_dir: str = "./cardiology_pdfs", 
                 index_dir: str = "./rag_index_langchain",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 k_retrieval: int = 5): # Number of documents to retrieve
        """Initialize the LangChain RAG evaluator."""
        
        self.model_name = model_name
        self.pdfs_dir = Path(pdfs_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval # Store k for retrieval
        
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model_name=self.model_name, 
            temperature=0, 
            api_key=OPENAI_API_KEY,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=42
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Ensure index exists or build it
        if not self._is_index_valid():
            print("LangChain RAG index not found or invalid. Building...")
            self.load_and_index_documents()
        else:
            print("Loading existing LangChain RAG index.")
            self._load_index()
            self._initialize_qa_chain()
    
    def _is_index_valid(self) -> bool:
        """Check if the ChromaDB index exists and is not empty."""
        if not self.index_dir.exists():
            return False
        # A simple check, ChromaDB creates multiple files.
        # Checking for specific files like 'chroma.sqlite3' or contents might be more robust.
        return any(self.index_dir.iterdir())
    
    def _load_index(self):
        """Load an existing vector store."""
        if self._is_index_valid():
            self.vector_store = Chroma(persist_directory=str(self.index_dir), embedding_function=self.embeddings)
            print(f"LangChain RAG index loaded from {self.index_dir}. Documents: {self.vector_store._collection.count()}")
        else:
            print(f"No valid index found at {self.index_dir}. Please build the index first.")
            raise FileNotFoundError(f"LangChain RAG index not found at {self.index_dir}")
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain."""
        if self.vector_store:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", # "stuff" is common, "map_reduce" or "refine" for large contexts
                retriever=self.vector_store.as_retriever(search_kwargs={"k": self.k_retrieval}),
                return_source_documents=True
            )
        else:
            print("Vector store not initialized. Cannot create QA chain.")
    
    def load_and_index_documents(self, pdf_glob_pattern: str = "*.pdf") -> int:
        """Load documents from PDF files and build/update the vector store."""
        print(f"Scanning for PDF files in {self.pdfs_dir} with pattern '{pdf_glob_pattern}'...")
        pdf_files = list(self.pdfs_dir.glob(pdf_glob_pattern))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdfs_dir} matching '{pdf_glob_pattern}'.")
            # Create an empty index if it doesn't exist, to prevent errors on load
            if not self.index_dir.exists():
                 self.index_dir.mkdir(parents=True, exist_ok=True)
                 # Create a dummy Chroma store to satisfy _is_index_valid in some cases
                 # This is a bit of a hack, ideally ChromaDB handles empty stores gracefully.
                 # For now, we ensure the directory exists.
                 Chroma.from_documents(documents=[Document(page_content="dummy")], embedding=self.embeddings, persist_directory=str(self.index_dir))
                 print(f"Created an empty index directory at {self.index_dir} as no PDFs were found.")
            self._load_index() # Attempt to load the (potentially empty) index
            self._initialize_qa_chain() # Initialize QA chain even if empty
            return 0

        print(f"Found {len(pdf_files)} PDF files. Loading and splitting documents...")
        
        all_docs = []
        for pdf_file in pdf_files:
            try:
                print(f"  Loading {pdf_file.name}...")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                all_docs.extend(documents)
            except Exception as e:
                print(f"  Error loading {pdf_file.name}: {e}")
        
        if not all_docs:
            print("No documents could be loaded from the PDF files.")
            # Similar handling for empty index as above
            if not self.index_dir.exists():
                self.index_dir.mkdir(parents=True, exist_ok=True)
                Chroma.from_documents(documents=[Document(page_content="dummy")], embedding=self.embeddings, persist_directory=str(self.index_dir))
                print(f"Created an empty index directory at {self.index_dir} as no documents were loaded.")
            self._load_index()
            self._initialize_qa_chain()
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        split_docs = text_splitter.split_documents(all_docs)
        
        print(f"Total documents loaded: {len(all_docs)}")
        print(f"Total chunks created: {len(split_docs)}")
        
        if not split_docs:
            print("No text chunks were created after splitting. Cannot build index.")
            # Similar handling for empty index
            if not self.index_dir.exists():
                self.index_dir.mkdir(parents=True, exist_ok=True)
                Chroma.from_documents(documents=[Document(page_content="dummy")], embedding=self.embeddings, persist_directory=str(self.index_dir))
                print(f"Created an empty index directory at {self.index_dir} as no text chunks were created.")
            self._load_index()
            self._initialize_qa_chain()
            return 0
            
        print(f"Building vector store at {self.index_dir} (this may take a while)...")
        start_time = time.time()
        
        # Delete existing index if it exists to rebuild fresh
        if self.index_dir.exists():
            print(f"  Deleting existing index at {self.index_dir} to rebuild...")
            import shutil
            shutil.rmtree(self.index_dir)
            
        self.vector_store = Chroma.from_documents(
            documents=split_docs, 
            embedding=self.embeddings,
            persist_directory=str(self.index_dir)
        )
        self.vector_store.persist() # Ensure data is written to disk
        
        end_time = time.time()
        print(f"Vector store built and persisted in {end_time - start_time:.2f} seconds.")
        print(f"Total chunks indexed: {self.vector_store._collection.count()}")

        self._initialize_qa_chain() # Initialize QA chain after building
        return len(split_docs)
    
    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if not self.qa_chain:
            print("QA chain not initialized. Cannot query. Ensure index is built/loaded.")
            # Try to initialize it if vector_store exists
            if self.vector_store:
                self._initialize_qa_chain()
                if not self.qa_chain: # Still not initialized
                     raise ValueError("QA chain could not be initialized. Vector store might be problematic.")
            else:
                raise ValueError("Vector store not available. Cannot initialize QA chain.")
        
        print(f"\n💬 Querying LangChain RAG: {question}")
        start_time = time.time()
        response = self.qa_chain({"query": question})
        end_time = time.time()
        
        print(f"  RAG Query completed in {end_time - start_time:.2f} seconds.")
        
        # response structure: {'query': '...', 'result': '...', 'source_documents': [Document(...)]}
        return {
            "answer": response["result"],
            "source_documents": [doc.page_content for doc in response["source_documents"]],
            "metadata": [doc.metadata for doc in response["source_documents"]]
        }
    
    def retrieve_context(self, question: str, k: Optional[int] = None) -> str:
        """Retrieve relevant context for a given question."""
        if not self.vector_store:
            print("Vector store not initialized. Cannot retrieve context. Ensure index is built/loaded.")
            # Try to load index if not loaded.
            if self._is_index_valid():
                self._load_index()
            else: # If still no vector store, then we must build.
                 print("Attempting to build index as it's required for context retrieval.")
                 self.load_and_index_documents()

            if not self.vector_store: # If after all attempts, still no vector store
                raise ValueError("Vector store could not be initialized. Cannot retrieve context.")

        print(f"\n Retrieving context for LangChain RAG: {question}")
        start_time = time.time()
        
        num_to_retrieve = k if k is not None else self.k_retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": num_to_retrieve})
        documents = retriever.get_relevant_documents(question)
        
        # Sort documents deterministically by content hash for reproducible results
        # This ensures that when multiple documents have similar similarity scores,
        # they are returned in a consistent order across runs
        documents = sorted(documents, key=lambda doc: hash(doc.page_content))
        
        end_time = time.time()
        print(f"  Context retrieval completed in {end_time - start_time:.2f} seconds. Retrieved {len(documents)} documents.")
        
        if not documents:
            return "No relevant context found in the knowledge base for this question."
            
        # Combine the content of the retrieved documents
        context_parts = []
        for i, doc in enumerate(documents):
            source_info = doc.metadata.get('source', 'Unknown source')
            page_info = doc.metadata.get('page', 'N/A')
            context_parts.append(f"Source: {source_info} (Page: {page_info})\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.vector_store:
            return {
                "implementation": "ChromaDB",
                "total_vectors": self.vector_store._collection.count(),
                "index_directory": str(self.index_dir.resolve())
            }
        return {"error": "Vector store not initialized."}

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Running LangChainRAGEvaluator standalone test...")
    
    # Create dummy PDFs if they don't exist for testing
    dummy_pdf_dir = Path("./dummy_cardiology_pdfs")
    dummy_pdf_dir.mkdir(exist_ok=True)
    if not list(dummy_pdf_dir.glob("*.pdf")):
        from reportlab.pdfgen import canvas
        def create_dummy_pdf(filename, content):
            c = canvas.Canvas(str(dummy_pdf_dir / filename))
            textobject = c.beginText(50, 750)
            for line in content.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            c.save()
        create_dummy_pdf("dummy_doc1.pdf", "This is about myocardial infarction and ECG changes.\nST elevation is key.")
        create_dummy_pdf("dummy_doc2.pdf", "Heart failure treatment often involves ACE inhibitors and beta-blockers.\nDiuretics can also be used.")
        print(f"Created dummy PDFs in {dummy_pdf_dir}")

    rag_evaluator = LangChainRAGEvaluator(pdfs_dir=str(dummy_pdf_dir), index_dir="./rag_index_langchain_test", k_retrieval=2)
    
    # Test index building (should happen in __init__ if not present)
    print("\n--- Index Stats ---")
    print(rag_evaluator.get_retrieval_stats())

    # Test context retrieval
    print("\n--- Context Retrieval Test ---")
    test_question_context = "What are the treatments for heart failure?"
    context = rag_evaluator.retrieve_context(test_question_context)
    print(f"Question: {test_question_context}")
    print(f"Retrieved Context:\n{context}")
    
    # Test querying
    print("\n--- Query Test ---")
    test_question_query = "What does ST elevation on an ECG indicate?"
    response = rag_evaluator.query(test_question_query)
    print(f"Question: {test_question_query}")
    print(f"Answer: {response['answer']}")
    print(f"Source Documents: {len(response['source_documents'])}")
    for i, (src_doc, meta) in enumerate(zip(response['source_documents'], response['metadata'])):
        print(f"  Source {i+1} (Page {meta.get('page', 'N/A')} from {meta.get('source', 'Unknown')}):\n    {src_doc[:200]}...") # Print snippet

    # Clean up dummy files and test index
    import shutil
    # shutil.rmtree(dummy_pdf_dir)
    # shutil.rmtree("./rag_index_langchain_test")
    # print("\nCleaned up dummy PDFs and test index.")
    print(f"\nTest complete. You may want to manually clean up: {dummy_pdf_dir} and ./rag_index_langchain_test")