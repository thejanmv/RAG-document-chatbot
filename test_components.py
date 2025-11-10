"""
Component Testing Script
Test individual components of the RAG system
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_gemini_connection():
    """Test Google Gemini API connection"""
    print("\n" + "="*50)
    print("Testing Gemini API Connection")
    print("="*50)
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            print("❌ GEMINI_API_KEY not set properly in .env")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content="Test embedding",
            task_type="retrieval_document"
        )
        
        if result and 'embedding' in result:
            print(f"✅ Gemini API connected successfully")
            print(f"   Embedding dimension: {len(result['embedding'])}")
            return True
        else:
            print("❌ Unexpected response from Gemini API")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Gemini: {str(e)}")
        return False

def test_pinecone_connection():
    """Test Pinecone connection"""
    print("\n" + "="*50)
    print("Testing Pinecone Connection")
    print("="*50)
    
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key or api_key.startswith("your_"):
            print("❌ PINECONE_API_KEY not set properly in .env")
            return False
        
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes()
        print(f"✅ Pinecone connected successfully")
        print(f"   Existing indexes: {[idx.name for idx in indexes]}")
        
        return True
            
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {str(e)}")
        return False

def test_document_processing():
    """Test document processing capabilities"""
    print("\n" + "="*50)
    print("Testing Document Processing")
    print("="*50)
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Test text splitting
        text = "This is a test. " * 100
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        chunks = splitter.split_text(text)
        print(f"✅ Text splitting works")
        print(f"   Created {len(chunks)} chunks from test text")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing document processing: {str(e)}")
        return False

def test_file_readers():
    """Test file reading libraries"""
    print("\n" + "="*50)
    print("Testing File Readers")
    print("="*50)
    
    readers = {
        'PyPDF2': 'PDF files',
        'docx2txt': 'DOCX files',
        'pandas': 'CSV files'
    }
    
    all_ok = True
    for module, file_type in readers.items():
        try:
            __import__(module)
            print(f"✅ {module} installed - Can read {file_type}")
        except ImportError:
            print(f"❌ {module} not installed - Cannot read {file_type}")
            all_ok = False
    
    return all_ok

def test_full_workflow():
    """Test a minimal end-to-end workflow"""
    print("\n" + "="*50)
    print("Testing End-to-End Workflow")
    print("="*50)
    
    try:
        # Test embedding generation
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        test_text = "Artificial Intelligence is transforming how we work and live."
        
        # Generate embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content=test_text,
            task_type="retrieval_document"
        )
        embedding = result['embedding']
        print(f"✅ Generated embedding for test text")
        
        # Test generation
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Say 'Hello, RAG system working!' and nothing else.")
        print(f"✅ Generated response: {response.text.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow test: {str(e)}")
        return False

def main():
    """Run all component tests"""
    print("\n" + "="*60)
    print("  RAG Document Assistant - Component Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Gemini Connection", test_gemini_connection()))
    results.append(("Pinecone Connection", test_pinecone_connection()))
    results.append(("Document Processing", test_document_processing()))
    results.append(("File Readers", test_file_readers()))
    results.append(("Full Workflow", test_full_workflow()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Upload a document")
        print("  3. Start asking questions!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Verify API keys in .env file")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check internet connection")
    
    print("="*60)

if __name__ == "__main__":
    main()

