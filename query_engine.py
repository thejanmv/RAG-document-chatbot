"""
Query Engine Module for RAG-Powered Document Assistant

This module handles:
- Query embedding generation using OpenAI
- Semantic retrieval from Pinecone
- Context-aware answer generation using OpenAI LLM
"""

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI

# Pinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
TOP_K = 3  # Number of chunks to retrieve


class QueryEngine:
    """Handles query processing, retrieval, and answer generation"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the query engine
        
        Args:
            model_name: Name of the OpenAI model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
        """
        self.embedding_model = EMBEDDING_MODEL
        self.model_name = model_name
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = None
        
        # Connect to index
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Connect to Pinecone index"""
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error connecting to Pinecone index: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query using OpenAI
        
        Args:
            query: User's question
            
        Returns:
            Query embedding vector
        """
        try:
            response = openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks from Pinecone
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Query Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            chunks = []
            for match in results['matches']:
                chunks.append({
                    "id": match['id'],
                    "score": match['score'],
                    "text": match['metadata'].get('text', ''),
                    "source": match['metadata'].get('source', 'Unknown'),
                    "chunk_index": match['metadata'].get('chunk_index', 0)
                })
            
            return chunks
        except Exception as e:
            print(f"Error retrieving from Pinecone: {str(e)}")
            raise
    
    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for the LLM with retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['source']
            text = chunk['text']
            context_text += f"\n[Document {i}: {source}]\n{text}\n"
        
        # Build the full prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Context from documents:
{context_text}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise and accurate
- Cite which document(s) you're referencing when appropriate
- If you're making any inferences, make that clear

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using OpenAI LLM
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer
        """
        # Build prompt
        prompt = self.build_prompt(query, context_chunks)
        
        # Generate response
        try:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided document context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Main query function: retrieve and generate answer
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing answer and retrieved chunks
        """
        try:
            # Retrieve relevant chunks
            print(f"Retrieving top {top_k} relevant chunks...")
            chunks = self.retrieve_relevant_chunks(question, top_k)
            
            if not chunks:
                return {
                    "answer": "No relevant information found in the uploaded documents.",
                    "chunks": [],
                    "error": None
                }
            
            print(f"Retrieved {len(chunks)} chunks")
            
            # Generate answer
            print("Generating answer with Gemini...")
            answer = self.generate_answer(question, chunks)
            
            return {
                "answer": answer,
                "chunks": chunks,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "answer": None,
                "chunks": [],
                "error": error_msg
            }
    
    def get_index_status(self) -> Dict[str, Any]:
        """
        Get information about the current Pinecone index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 768),
                "index_fullness": stats.get('index_fullness', 0.0)
            }
        except Exception as e:
            return {
                "error": str(e)
            }


class EvaluationMetrics:
    """Optional: Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def calculate_retrieval_metrics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate simple metrics about retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Dictionary with metrics
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_score": 0.0,
                "sources": []
            }
        
        scores = [chunk['score'] for chunk in chunks]
        sources = list(set([chunk['source'] for chunk in chunks]))
        
        return {
            "num_chunks": len(chunks),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "sources": sources,
            "num_sources": len(sources)
        }


def test_query_engine(question: str, model_name: str = DEFAULT_LLM_MODEL):
    """
    Test function for the query engine
    
    Args:
        question: Test question
        model_name: Gemini model to use
    """
    print(f"\n{'='*50}")
    print(f"Testing Query Engine")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    # Initialize query engine
    engine = QueryEngine(model_name=model_name)
    
    # Get index status
    status = engine.get_index_status()
    print(f"Index Status: {status}")
    
    # Query
    result = engine.query(question)
    
    if result['error']:
        print(f"Error: {result['error']}")
        return
    
    # Display results
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(result['answer'])
    
    print("\n" + "="*50)
    print(f"RETRIEVED CHUNKS ({len(result['chunks'])}):")
    print("="*50)
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"\nChunk {i}:")
        print(f"  Source: {chunk['source']}")
        print(f"  Score: {chunk['score']:.4f}")
        print(f"  Text: {chunk['text'][:200]}...")
    
    # Calculate metrics
    metrics = EvaluationMetrics.calculate_retrieval_metrics(result['chunks'])
    print("\n" + "="*50)
    print("METRICS:")
    print("="*50)
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Test with a sample question
    test_question = "What are the main topics discussed in the documents?"
    
    print("Query Engine Module")
    print("This module is meant to be imported by the Streamlit app")
    print("\nFor testing, you can run:")
    print(f'python -c "from query_engine import test_query_engine; test_query_engine(\'{test_question}\')"')
    print("\nOr run the full app with: streamlit run app.py")

