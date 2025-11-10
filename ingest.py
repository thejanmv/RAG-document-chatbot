"""
Document Ingestion Module for RAG-Powered Document Assistant

This module handles:
- Document loading (PDF, DOCX, CSV)
- Text extraction and chunking
- Embedding generation using OpenAI
- Storage in Pinecone vector database
"""

import os
import io
from typing import List, Dict, Any
from dotenv import load_dotenv

# Document processing
import PyPDF2
import docx2txt
import pandas as pd

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs")

# Text splitter configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class DocumentProcessor:
    """Handles document loading, chunking, and embedding generation"""
    
    def __init__(self):
        """Initialize the document processor"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize LangChain OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            # docx2txt requires file path or file-like object
            text = docx2txt.process(file)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_text_from_csv(self, file) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file)
            # Convert DataFrame to text format
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from CSV: {str(e)}")
    
    def load_document(self, file, filename: str) -> str:
        """Load and extract text from document based on file type"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file)
        elif file_extension == 'csv':
            return self.extract_text_from_csv(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def chunk_text(self, text: str, filename: str) -> List[Document]:
        """Split text into chunks using RecursiveCharacterTextSplitter"""
        # Create a Document object with metadata
        doc = Document(page_content=text, metadata={"source": filename})
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI embedding model via LangChain"""
        try:
            # Generate embeddings using LangChain's OpenAI wrapper
            # This handles batching, retries, and rate limiting automatically
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            error_msg = str(e)
            # Check for specific quota/rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower() or "rate_limit" in error_msg.lower():
                raise Exception(
                    f"❌ OpenAI API Rate Limit/Quota Exceeded!\n\n"
                    f"You've hit the rate limit for embedding requests.\n\n"
                    f"Solutions:\n"
                    f"1. Wait a minute and try again\n"
                    f"2. Check usage at: https://platform.openai.com/usage\n"
                    f"3. Upgrade your plan at: https://platform.openai.com/account/billing\n"
                    f"4. Add more credits to your account\n\n"
                    f"Total chunks to process: {len(texts)}"
                )
            else:
                print(f"Error generating embeddings: {error_msg}")
                raise Exception(f"Embedding generation failed: {error_msg[:200]}")
    
    def process_document(self, file, filename: str) -> List[Dict[str, Any]]:
        """
        Process a document: extract text, chunk, and generate embeddings
        
        Returns:
            List of dictionaries containing chunk text, embedding, and metadata
        """
        # Extract text
        print(f"Extracting text from {filename}...")
        text = self.load_document(file, filename)
        
        if not text or len(text.strip()) == 0:
            raise ValueError(f"No text extracted from {filename}")
        
        print(f"Extracted {len(text)} characters")
        
        # Chunk text
        print("Chunking text...")
        chunks = self.chunk_text(text, filename)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings with Gemini...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)
        
        # Combine chunks with embeddings
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                "id": f"{filename}_{i}",
                "text": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata
            })
        
        return processed_chunks


class PineconeManager:
    """Manages Pinecone vector database operations"""
    
    def __init__(self):
        """Initialize Pinecone client"""
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        self.index = None
    
    def create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Index {self.index_name} already exists")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def store_embeddings(self, processed_chunks: List[Dict[str, Any]]):
        """Store embeddings in Pinecone"""
        if not self.index:
            self.create_index_if_not_exists()
        
        # Prepare vectors for upsert
        vectors = []
        for chunk in processed_chunks:
            vectors.append({
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"]
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")
        
        print(f"Successfully stored {len(vectors)} embeddings in Pinecone")
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index"""
        if not self.index:
            self.create_index_if_not_exists()
        
        return self.index.describe_index_stats()


def ingest_documents(files_with_names: List[tuple]) -> Dict[str, Any]:
    """
    Main function to ingest documents into the RAG system
    
    Args:
        files_with_names: List of tuples (file_object, filename)
    
    Returns:
        Dictionary with ingestion statistics
    """
    processor = DocumentProcessor()
    pinecone_manager = PineconeManager()
    
    # Ensure Pinecone index exists
    pinecone_manager.create_index_if_not_exists()
    
    total_chunks = 0
    results = []
    
    for file, filename in files_with_names:
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print(f"{'='*50}")
            
            # Process document
            processed_chunks = processor.process_document(file, filename)
            
            # Store in Pinecone
            pinecone_manager.store_embeddings(processed_chunks)
            
            total_chunks += len(processed_chunks)
            results.append({
                "filename": filename,
                "chunks": len(processed_chunks),
                "status": "success"
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            results.append({
                "filename": filename,
                "chunks": 0,
                "status": "error",
                "error": str(e)
            })
    
    # Get final index stats
    stats = pinecone_manager.get_index_stats()
    
    return {
        "total_documents": len(files_with_names),
        "total_chunks": total_chunks,
        "results": results,
        "index_stats": stats
    }


if __name__ == "__main__":
    # Test the ingestion with a sample file
    print("Document Ingestion Module")
    print("This module is meant to be imported by the Streamlit app")
    print("Run: streamlit run app.py")

