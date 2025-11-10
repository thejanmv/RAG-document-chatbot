"""
Streamlit App for RAG-Powered Document Assistant (OpenAI Version)

A web interface for uploading documents and asking questions using
Retrieval-Augmented Generation with OpenAI GPT models.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO

# Import our modules
from ingest import ingest_documents, PineconeManager
from query_engine import QueryEngine, EvaluationMetrics

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chunk-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False


def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please create a .env file with the required API keys. See env.example for reference.")
        return False
    return True


def display_header():
    """Display the app header"""
    st.markdown('<div class="main-header">📚 OpenAI-Powered RAG Document Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload documents and ask questions using AI-powered retrieval</div>', unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with settings and information"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Select OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
            index=0,
            help="gpt-4o-mini is fast & cheap, gpt-4o is most capable"
        )
        
        # Number of chunks to retrieve
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="More chunks provide more context but may slow down response"
        )
        
        st.divider()
        
        # Index information
        st.header("📊 Index Status")
        if st.button("Refresh Index Stats"):
            try:
                pm = PineconeManager()
                pm.create_index_if_not_exists()
                stats = pm.get_index_stats()
                
                total_vectors = stats.get('total_vector_count', 0)
                st.metric("Total Vectors", total_vectors)
                
                if total_vectors > 0:
                    st.success("Index is ready!")
                else:
                    st.info("No documents uploaded yet")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Information
        st.header("ℹ️ About")
        st.markdown("""
        This app uses:
        - **OpenAI GPT** for embeddings and generation
        - **Pinecone** for vector storage
        - **LangChain** for document processing
        
        Supported formats:
        - PDF (.pdf)
        - Word (.docx)
        - CSV (.csv)
        """)
        
        # Clear chat history
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        return model_name, top_k


def process_uploaded_files(uploaded_files):
    """Process uploaded files and ingest into vector database"""
    if not uploaded_files:
        st.warning("Please upload at least one file")
        return False
    
    with st.spinner("Processing documents..."):
        try:
            # Prepare files for ingestion
            files_with_names = []
            for uploaded_file in uploaded_files:
                # Create a BytesIO object from uploaded file
                file_bytes = BytesIO(uploaded_file.read())
                files_with_names.append((file_bytes, uploaded_file.name))
            
            # Ingest documents
            results = ingest_documents(files_with_names)
            
            # Display results
            st.success(f"Successfully processed {results['total_documents']} document(s)!")
            
            # Show details
            with st.expander("View Processing Details"):
                for result in results['results']:
                    if result['status'] == 'success':
                        st.write(f"✅ **{result['filename']}**: {result['chunks']} chunks")
                    else:
                        st.write(f"❌ **{result['filename']}**: {result.get('error', 'Unknown error')}")
                
                st.write(f"\n**Total chunks created**: {results['total_chunks']}")
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False


def display_chat_interface(model_name, top_k):
    """Display the chat interface for asking questions"""
    st.header("💬 Ask Questions")
    
    # Initialize query engine if not already done
    if st.session_state.query_engine is None or st.session_state.query_engine.model_name != model_name:
        try:
            with st.spinner("Initializing query engine..."):
                st.session_state.query_engine = QueryEngine(model_name=model_name)
        except Exception as e:
            st.error(f"Error initializing query engine: {str(e)}")
            return
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            if chat.get('show_chunks', False):
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(chat['chunks'], 1):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <strong>Chunk {i}</strong> - {chunk['source']}<br>
                            <small>Relevance Score: {chunk['score']:.4f}</small><br>
                            <p style="margin-top: 0.5rem;">{chunk['text'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Question input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="What are the main topics discussed in the documents?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Show chunks toggle
    show_chunks = st.checkbox("Show retrieved context", value=False)
    
    # Process question
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                # Query the engine
                result = st.session_state.query_engine.query(question, top_k=top_k)
                
                if result['error']:
                    st.error(result['error'])
                    return
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'chunks': result['chunks'],
                    'show_chunks': show_chunks
                })
                
                # Rerun to display new message
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")


def main():
    """Main application function"""
    # Check environment
    if not check_environment():
        st.stop()
    
    # Display header
    display_header()
    
    # Display sidebar and get settings
    model_name, top_k = display_sidebar()
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])
    
    with tab1:
        st.header("Upload Your Documents")
        st.write("Upload one or more documents to get started. Supported formats: PDF, DOCX, CSV")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload documents to create a searchable knowledge base"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Process Documents", type="primary", disabled=not uploaded_files):
                if process_uploaded_files(uploaded_files):
                    st.session_state.documents_processed = True
                    st.balloons()
        
        with col2:
            if st.button("Clear Uploaded Files"):
                uploaded_files = None
                st.rerun()
        
        # Display uploaded files
        if uploaded_files:
            st.write(f"\n**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # KB
                st.write(f"- {file.name} ({file_size:.1f} KB)")
    
    with tab2:
        # Check if documents have been processed
        try:
            pm = PineconeManager()
            pm.create_index_if_not_exists()
            stats = pm.get_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                st.info("👈 Please upload and process documents first in the 'Upload Documents' tab")
            else:
                display_chat_interface(model_name, top_k)
                
        except Exception as e:
            st.error(f"Error checking index status: {str(e)}")


if __name__ == "__main__":
    main()

